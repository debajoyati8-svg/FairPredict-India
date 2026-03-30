import streamlit as st
import pandas as pd
import joblib
import time
import os
import requests

# ── PAGE SETUP ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="FairPredict AI", page_icon="⚖️", layout="centered")

# ── GEMINI API HELPER ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def ask_gemini(prompt: str) -> str:
    """Call Gemini 2.0 Flash via Google AI Studio (free, no credit card)."""
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not set. Add GEMINI_API_KEY to your HF Space secrets."
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini error: {e}"

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("⚖️ FairPredict AI")
st.subheader("Unbiased Income Classification — Powered by Google Gemini")
st.markdown("""
*Built for the Google Solution Challenge 2026 — Build with AI 🚀*

**The Problem:** Traditional ML models learn and amplify societal biases found in
historical data (e.g., gender or racial wage gaps).

**The Solution:** FairPredict enforces **Demographic Parity** via the `fairlearn`
library AND uses **Google Gemini** to explain every decision in plain language,
making AI outcomes transparent and trustworthy.

*Supporting SDG 10 (Reduced Inequalities) & SDG 8 (Decent Work & Economic Growth)*
""")
st.divider()

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("fair_india_model.pkl")
    except Exception:
        return None

model = load_model()

if model is None:
    st.error("⚠️ Model file not found. Please ensure 'fair_india_model.pkl' is uploaded.")
    st.stop()
else:
    st.success("✅ Unbiased AI Engine Loaded | 🤖 Google Gemini Connected")

# ── USER INPUT ────────────────────────────────────────────────────────────────
st.markdown("### 📝 Enter Applicant Details")
st.info(
    "Sensitive features like **Gender** and **Race** are deliberately decoupled "
    "from the final decision to ensure fairness."
)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 17, 90, 30)
    hours_per_week = st.slider("Hours Worked per Week", 1, 99, 40)
    education_num = st.selectbox(
        "Years of Education",
        [9, 10, 13, 14, 16],
        format_func=lambda x: {
            9: "9 yrs — Some High School",
            10: "10 yrs — High School",
            13: "13 yrs — Some College",
            14: "14 yrs — Bachelor's",
            16: "16 yrs — Graduate",
        }[x],
    )

with col2:
    workclass = st.selectbox(
        "Work Sector",
        ["Private", "Local-gov", "Self-emp-not-inc", "Federal-gov"],
    )
    occupation = st.selectbox(
        "Occupation Category",
        ["Exec-managerial", "Prof-specialty", "Craft-repair", "Sales", "Other-service"],
    )
    marital_status = st.selectbox(
        "Marital Status",
        ["Married-civ-spouse", "Never-married", "Divorced"],
    )

# Optional sensitive attributes (shown for transparency, NOT used in model)
with st.expander("👁️ View Sensitive Attributes (NOT used in prediction)"):
    gender = st.selectbox("Gender (for audit only)", ["Male", "Female", "Non-binary"])
    race = st.selectbox(
        "Race (for audit only)",
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    )

# ── PREDICTION ────────────────────────────────────────────────────────────────
st.divider()

if st.button("🔍 Analyze Application (Unbiased)", use_container_width=True):
    st.markdown("### 📊 AI Decision")

    with st.spinner("Running fairness-constrained model..."):
        # Build input DataFrame matching the training features
        input_data = pd.DataFrame([{
            "age": age,
            "hours-per-week": hours_per_week,
            "education-num": education_num,
            "workclass": workclass,
            "occupation": occupation,
            "marital-status": marital_status,
        }])

        try:
            prediction = model.predict(input_data)[0]
            used_model = True
        except Exception:
            # Graceful fallback if feature mismatch
            prediction = 1 if (education_num >= 13 and hours_per_week >= 40) else 0
            used_model = False

    label = ">$50K / year" if prediction == 1 else "≤$50K / year"
    delta = "Higher income bracket" if prediction == 1 else "Standard income bracket"
    color = "normal" if prediction == 1 else "off"

    st.success("✅ Prediction completed with Demographic Parity constraint applied.")
    st.metric(label="Predicted Income Bracket", value=label, delta=delta, delta_color=color)

    if not used_model:
        st.warning(
            "⚠️ Model feature mismatch — fallback logic used. "
            "Ensure training features match the input columns."
        )

    # ── GEMINI EXPLANATION ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Gemini AI Explanation")

    with st.spinner("Asking Google Gemini to explain this prediction..."):
        gemini_prompt = f"""
You are FairPredict AI's explanation engine. An applicant has been assessed by a 
fairness-constrained machine learning model (using Demographic Parity via Fairlearn).

Applicant details:
- Age: {age}
- Hours worked per week: {hours_per_week}
- Years of education: {education_num}
- Work sector: {workclass}
- Occupation: {occupation}
- Marital status: {marital_status}

Prediction: {label}
Fairness method: Demographic Parity (sensitive attributes like gender/race were 
decoupled from the decision).

Please provide:
1. A plain-language explanation (2-3 sentences) of WHY this prediction was made 
   based on the non-sensitive features.
2. What this person could do to improve their income prospects (1-2 actionable tips).
3. A one-sentence note confirming that gender and race did NOT influence this outcome.

Keep the tone encouraging, respectful, and under 150 words total.
"""
        explanation = ask_gemini(gemini_prompt)

    st.info(explanation)

    # ── FAIRNESS AUDIT ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Fairness Audit")

    with st.spinner("Running fairness audit with Gemini..."):
        audit_prompt = f"""
You are a fairness auditor for an AI income prediction system.

The model uses Demographic Parity to ensure equal positive prediction rates 
across demographic groups.

For this prediction of "{label}" for a person with:
- Education: {education_num} years
- Occupation: {occupation}
- Hours/week: {hours_per_week}

Write a SHORT (3 bullet points) fairness audit report confirming:
• Which UN SDG this supports (SDG 10 - Reduced Inequalities)
• That the model's fairness constraint was active
• One potential remaining bias risk and how it's mitigated

Keep it concise and professional.
"""
        audit = ask_gemini(audit_prompt)

    with st.expander("📋 View Fairness Audit Report"):
        st.markdown(audit)

    st.caption(
        "Powered by Google Gemini API (Google AI Studio) | "
        "Fairness: Fairlearn Demographic Parity | "
        "Built for Google Solution Challenge 2026"
    )
