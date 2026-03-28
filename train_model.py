"""
FairPredict India - Model Training Script
Generates realistic Indian income data and trains a fair ML model.
Run this locally to generate: fair_india_model.pkl
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 10000

# ── Indian states grouped by development tier ──────────────────────────────
tier1_states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Telangana']
tier2_states = ['Gujarat', 'Andhra Pradesh', 'Rajasthan', 'Kerala', 'Punjab']
tier3_states = ['Uttar Pradesh', 'Bihar', 'Jharkhand', 'Odisha', 'Assam']

state_choices = tier1_states * 3 + tier2_states * 2 + tier3_states
state = np.random.choice(state_choices, N)
state_tier = np.where(np.isin(state, tier1_states), 2,
             np.where(np.isin(state, tier2_states), 1, 0))

# ── Demographics ────────────────────────────────────────────────────────────
age = np.random.randint(18, 65, N)

gender = np.random.choice(['Male', 'Female'], N, p=[0.60, 0.40])

# Indian caste categories (sensitive feature for fairness)
caste = np.random.choice(
    ['General', 'OBC', 'SC', 'ST'],
    N, p=[0.30, 0.40, 0.20, 0.10]
)

# ── Education (Indian system) ───────────────────────────────────────────────
education_map = {
    'Below 10th':    0,
    '10th Pass':     1,
    '12th Pass':     2,
    'Diploma':       3,
    'Graduate':      4,
    'Post-Graduate': 5,
    'Doctorate':     6,
}
edu_choices = list(education_map.keys())
edu_probs   = [0.10, 0.15, 0.20, 0.10, 0.30, 0.12, 0.03]
education   = np.random.choice(edu_choices, N, p=edu_probs)
edu_num     = np.array([education_map[e] for e in education])

# ── Work sector (Indian) ────────────────────────────────────────────────────
sector = np.random.choice(
    ['Private', 'Government', 'Self-Employed', 'Agriculture', 'Informal'],
    N, p=[0.40, 0.15, 0.20, 0.15, 0.10]
)
sector_map = {'Private': 3, 'Government': 4, 'Self-Employed': 2,
              'Agriculture': 1, 'Informal': 0}
sector_num = np.array([sector_map[s] for s in sector])

# ── Occupation (Indian context) ─────────────────────────────────────────────
occupation = np.random.choice(
    ['IT/Software', 'Healthcare', 'Education', 'Manufacturing',
     'Finance/Banking', 'Agriculture', 'Construction', 'Retail/Trade',
     'Government Service', 'Other'],
    N, p=[0.12, 0.08, 0.10, 0.12, 0.08, 0.15, 0.10, 0.10, 0.10, 0.05]
)
occ_income_boost = {
    'IT/Software': 3, 'Finance/Banking': 3, 'Healthcare': 2,
    'Government Service': 2, 'Education': 1, 'Manufacturing': 1,
    'Retail/Trade': 0, 'Construction': 0, 'Agriculture': -1, 'Other': 0
}
occ_boost = np.array([occ_income_boost[o] for o in occupation])

# ── Hours & Experience ──────────────────────────────────────────────────────
hours_per_week  = np.random.randint(20, 80, N)
years_exp       = np.clip(age - 18 - (6 - edu_num), 0, 40)

# ── Family & assets ─────────────────────────────────────────────────────────
family_size = np.random.randint(1, 8, N)
has_loan    = np.random.choice([0, 1], N, p=[0.55, 0.45])

# ── Income label (₹5 LPA threshold = ~₹41,667/month) ───────────────────────
# Realistic probability function incorporating Indian factors
log_odds = (
    -3.0
    + 0.04  * (age - 30)
    + 0.60  * edu_num
    + 0.25  * sector_num
    + 0.40  * occ_boost
    + 0.015 * hours_per_week
    + 0.05  * years_exp
    + 0.30  * state_tier
    - 0.05  * family_size
    - 0.20  * has_loan
    # Intentional bias in raw data (fairlearn will correct this)
    + np.where(gender == 'Male', 0.40, 0.0)
    + np.where(caste == 'General', 0.35,
      np.where(caste == 'OBC', 0.10, -0.20))
)
prob = 1 / (1 + np.exp(-log_odds))
income_above_5lpa = (np.random.rand(N) < prob).astype(int)

# ── Build DataFrame ─────────────────────────────────────────────────────────
df = pd.DataFrame({
    'age':            age,
    'education':      education,
    'edu_num':        edu_num,
    'gender':         gender,
    'caste':          caste,
    'state':          state,
    'state_tier':     state_tier,
    'sector':         sector,
    'sector_num':     sector_num,
    'occupation':     occupation,
    'occ_boost':      occ_boost,
    'hours_per_week': hours_per_week,
    'years_exp':      years_exp,
    'family_size':    family_size,
    'has_loan':       has_loan,
    'income_above_5lpa': income_above_5lpa,
})

print(f"Dataset shape: {df.shape}")
print(f"Income >5LPA: {income_above_5lpa.mean()*100:.1f}%")
print(f"Male >5LPA:   {df[df.gender=='Male']['income_above_5lpa'].mean()*100:.1f}%")
print(f"Female >5LPA: {df[df.gender=='Female']['income_above_5lpa'].mean()*100:.1f}%")

# ── Feature prep ────────────────────────────────────────────────────────────
feature_cols = [
    'age', 'edu_num', 'sector_num', 'occ_boost',
    'hours_per_week', 'years_exp', 'state_tier',
    'family_size', 'has_loan'
]

X = df[feature_cols]
y = df['income_above_5lpa']
sensitive = df['gender']   # fairness enforced on gender

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.2, random_state=42
)

# ── Base model (biased) ──────────────────────────────────────────────────────
base_clf = GradientBoostingClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
)

# ── Fair model via ExponentiatedGradient + DemographicParity ────────────────
print("\nTraining fair model (this may take ~30 seconds)...")
mitigator = ExponentiatedGradient(
    estimator=base_clf,
    constraints=DemographicParity(),
    eps=0.02,          # fairness tolerance
    max_iter=30,
)
mitigator.fit(X_train, y_train, sensitive_features=s_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = mitigator.predict(X_test)
print(f"\nFair Model Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
print(classification_report(y_test, y_pred))

male_acc   = accuracy_score(y_test[s_test=='Male'],   y_pred[s_test=='Male'])
female_acc = accuracy_score(y_test[s_test=='Female'], y_pred[s_test=='Female'])
print(f"Male accuracy:   {male_acc*100:.1f}%")
print(f"Female accuracy: {female_acc*100:.1f}%")
print(f"Fairness gap:    {abs(male_acc - female_acc)*100:.2f}% (lower is better)")

# ── Save everything ──────────────────────────────────────────────────────────
model_bundle = {
    'model':        mitigator,
    'feature_cols': feature_cols,
    'occ_boost_map': occ_income_boost,
    'sector_map':   sector_map,
    'edu_map':      education_map,
}
joblib.dump(model_bundle, 'fair_india_model.pkl')
print("\n✅ Model saved as fair_india_model.pkl")
print("   Upload this file to your Hugging Face Space.")
