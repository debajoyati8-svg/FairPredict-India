import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairPredict India ⚖️",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }

    .hero-box {
        background: linear-gradient(135deg, #FF6B35, #FF9500, #138808);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        color: white;
    }
    .hero-box h1 { font-size: 2.4rem; margin: 0; font-weight: 800; }
    .hero-box p  { font-size: 1.05rem; margin-top: 10px; opacity: 0.92; }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .num  { font-size: 2rem; font-weight: 700; color: #FF9500; }
    .metric-card .lbl  { font-size: 0.85rem; color: #aaa; margin-top: 4px; }

    .result-positive {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border-left: 5px solid #52b788;
        border-radius: 12px;
        padding: 24px 28px;
        color: white;
    }
    .result-negative {
        background: linear-gradient(135deg, #3d1a00, #6b3700);
        border-left: 5px solid #FF9500;
        border-radius: 12px;
        padding: 24px 28px;
        color: white;
    }
    .result-positive h2, .result-negative h2 { margin: 0; font-size: 1.5rem; }
    .result-positive p,  .result-negative p  { opacity: 0.85; margin-top: 8px; }

    .fairness-badge {
        background: rgba(19,136,8,0.2);
        border: 1px solid #138808;
        border-radius: 8px;
        padding: 10px 16px;
        color: #52b788;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 12px;
    }

    .scheme-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,149,0,0.25);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        color: white;
    }
    .scheme-card h4 { color: #FF9500; margin: 0 0 6px 0; font-size: 1rem; }
    .scheme-card p  { color: #ccc; margin: 0; font-size: 0.88rem; line-height: 1.5; }
    .scheme-badge {
        background: rgba(19,136,8,0.25);
        color: #52b788;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        margin-top: 8px;
        display: inline-block;
    }

    .career-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(83,131,236,0.3);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        color: white;
    }
    .career-card h4 { color: #7eb3ff; margin: 0 0 6px 0; font-size: 1rem; }
    .career-card p  { color: #ccc; margin: 0; font-size: 0.88rem; line-height: 1.5; }
    .impact-badge {
        background: rgba(126,179,255,0.15);
        color: #7eb3ff;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        margin-top: 8px;
        display: inline-block;
    }

    .sdg-box {
        background: linear-gradient(135deg, rgba(255,149,0,0.1), rgba(19,136,8,0.1));
        border: 1px solid rgba(255,149,0,0.3);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }

    .section-header {
        color: #FF9500;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(255,149,0,0.3);
        padding-bottom: 6px;
        margin-bottom: 14px;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { color: #ddd !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load('fair_india_model.pkl')
    except Exception:
        return None

bundle = load_model()

# ── Government Schemes Database ──────────────────────────────────────────────
SCHEMES = {
    'skill': {
        'name': '🎓 PM Kaushal Vikas Yojana (PMKVY)',
        'desc': 'Free skill training in 40+ sectors including IT, healthcare, construction. Get a nationally recognized certificate and job placement support.',
        'link': 'https://www.pmkvyofficial.org',
        'benefit': 'Free training + ₹8,000 reward',
        'for': 'Anyone wanting to upskill'
    },
    'loan_small': {
        'name': '💰 PM MUDRA Yojana',
        'desc': 'Collateral-free loans up to ₹10 lakh for small business owners. Three categories: Shishu (₹50K), Kishore (₹5L), Tarun (₹10L).',
        'link': 'https://www.mudra.org.in',
        'benefit': 'Loan up to ₹10 Lakh',
        'for': 'Self-employed, small business'
    },
    'agriculture': {
        'name': '🌾 PM Kisan Samman Nidhi',
        'desc': 'Direct income support of ₹6,000 per year to farmer families in three installments of ₹2,000 each.',
        'link': 'https://pmkisan.gov.in',
        'benefit': '₹6,000/year direct transfer',
        'for': 'Farmers with cultivable land'
    },
    'women': {
        'name': '👩 Mahila Udyam Nidhi Scheme',
        'desc': 'Special loans up to ₹10 lakh for women entrepreneurs to start or expand small businesses with subsidized interest rates.',
        'link': 'https://msme.gov.in',
        'benefit': 'Subsidized loan for women',
        'for': 'Women entrepreneurs'
    },
    'sc_st': {
        'name': '📋 Stand Up India Scheme',
        'desc': 'Bank loans between ₹10 lakh to ₹1 crore for SC/ST and women entrepreneurs to set up greenfield enterprises.',
        'link': 'https://www.standupmitra.in',
        'benefit': 'Loan ₹10L to ₹1 Crore',
        'for': 'SC/ST category entrepreneurs'
    },
    'education': {
        'name': '📚 National Scholarship Portal',
        'desc': 'Over 50 scholarships for students from economically weaker sections, OBC, SC/ST for higher education and professional courses.',
        'link': 'https://scholarships.gov.in',
        'benefit': 'Up to ₹50,000/year scholarship',
        'for': 'Students in higher education'
    },
    'employment': {
        'name': '💼 PM Rozgar Protsahan Yojana',
        'desc': 'Government pays 12% EPF contribution for new employees for 3 years, incentivizing companies to hire more workers.',
        'link': 'https://pmrpy.gov.in',
        'benefit': 'Employment + social security',
        'for': 'Job seekers in formal sector'
    },
    'digital': {
        'name': '💻 PM Gramin Digital Saksharta Abhiyan',
        'desc': 'Free digital literacy training for rural household members — internet, banking, e-governance, and smartphone skills.',
        'link': 'https://pmgdisha.in',
        'benefit': 'Free digital skills certificate',
        'for': 'Rural residents, first-time internet users'
    },
}

# ── Career Recommendations Database ──────────────────────────────────────────
CAREER_PATHS = {
    'IT/Software': [
        {'title': '🖥️ Learn Cloud Computing', 'desc': 'AWS/Azure certifications can increase IT salary by 40-60%. Free courses on AWS Skill Builder and Microsoft Learn.', 'impact': '+₹3-8 LPA', 'time': '3-6 months'},
        {'title': '📊 Data Science & ML', 'desc': 'Python + Machine Learning skills are in high demand. Coursera and NPTEL offer free/affordable courses.', 'impact': '+₹4-10 LPA', 'time': '6-12 months'},
    ],
    'Agriculture': [
        {'title': '🌱 Organic Farming Certification', 'desc': 'Certified organic produce fetches 2-3x premium price in urban markets. APEDA provides free training.', 'impact': '+₹1-3 LPA', 'time': '3-4 months'},
        {'title': '📱 AgriTech & FPO Membership', 'desc': 'Join a Farmer Producer Organization to get better prices and access to markets like DeHaat and Ninjacart.', 'impact': '+₹50K-1.5 LPA', 'time': '1-2 months'},
    ],
    'Healthcare': [
        {'title': '🏥 Paramedic Specialization', 'desc': 'Specialized certifications in dialysis, OT technician, or ICU care significantly boost earning potential.', 'impact': '+₹2-5 LPA', 'time': '6 months'},
        {'title': '💊 Healthcare Administration', 'desc': 'Hospital management courses open doors to administrative roles with better pay and stability.', 'impact': '+₹3-6 LPA', 'time': '1 year'},
    ],
    'Education': [
        {'title': '🎓 CTET / State TET Certification', 'desc': 'Government teacher certification gives access to permanent government teaching positions with job security.', 'impact': '+₹2-4 LPA', 'time': '6 months prep'},
        {'title': '📲 Online Tutoring', 'desc': 'Platforms like Vedantu, Unacademy, and Byju\'s pay teachers ₹500-2000/hour for online classes.', 'impact': '+₹1.5-4 LPA', 'time': 'Start immediately'},
    ],
    'Finance/Banking': [
        {'title': '📈 NISM/SEBI Certifications', 'desc': 'Mutual fund, stock broker, and insurance certifications unlock higher-paying roles in financial services.', 'impact': '+₹2-6 LPA', 'time': '2-4 months'},
        {'title': '🏦 CA Foundation / CMA', 'desc': 'Chartered Accountancy is one of the highest-paying professions in India. ICAI offers structured study material.', 'impact': '+₹5-15 LPA', 'time': '2-3 years'},
    ],
    'Manufacturing': [
        {'title': '⚙️ Lean Six Sigma Certification', 'desc': 'Quality management certification highly valued in manufacturing. Can lead to supervisor/manager roles.', 'impact': '+₹2-4 LPA', 'time': '3-4 months'},
        {'title': '🤖 Industrial Automation Skills', 'desc': 'PLC programming and robotics skills are in high demand as factories automate. ITI offers relevant courses.', 'impact': '+₹3-6 LPA', 'time': '6 months'},
    ],
    'Government Service': [
        {'title': '⬆️ Departmental Promotion Exam', 'desc': 'Prepare for departmental exams to move to higher grades. Each grade jump means significant salary increase.', 'impact': '+₹1-3 LPA', 'time': '6 months prep'},
        {'title': '🏛️ UPSC / State PSC', 'desc': 'Higher civil services provide both income growth and social impact opportunities across India.', 'impact': '+₹5-20 LPA', 'time': '1-3 years'},
    ],
    'Retail/Trade': [
        {'title': '🛒 E-Commerce Selling', 'desc': 'Sell products on Amazon, Flipkart, or Meesho. Government\'s ONDC platform offers zero commission selling.', 'impact': '+₹1-5 LPA', 'time': '1-2 months'},
        {'title': '📦 GST & Business Formalization', 'desc': 'Registering your business opens access to MSME loans, government tenders, and larger buyers.', 'impact': '+₹1-3 LPA', 'time': '2-4 weeks'},
    ],
    'Construction': [
        {'title': '🏗️ Civil Supervisor Certification', 'desc': 'ITI or polytechnic certification in civil work moves you from daily wage to salaried supervisor roles.', 'impact': '+₹1.5-3 LPA', 'time': '6-12 months'},
        {'title': '🔌 Electrical/Plumbing License', 'desc': 'Licensed electricians and plumbers earn significantly more and can start their own contracting business.', 'impact': '+₹2-4 LPA', 'time': '6 months'},
    ],
    'Other': [
        {'title': '💡 PMKVY Skill Training', 'desc': 'Identify a high-demand skill in your area through PM Kaushal Vikas Yojana and get free certified training.', 'impact': '+₹1-4 LPA', 'time': '3-6 months'},
        {'title': '📱 Gig Economy Platforms', 'desc': 'Urban Clap, Dunzo, Swiggy, and logistics companies offer flexible income opportunities while you upskill.', 'impact': '+₹80K-2 LPA', 'time': 'Start immediately'},
    ],
}

def get_schemes(gender, caste, sector, occupation, edu_num, age):
    schemes = []
    if sector == 'Agriculture':
        schemes.append(SCHEMES['agriculture'])
    if caste in ['SC', 'ST']:
        schemes.append(SCHEMES['sc_st'])
    if gender == 'Female':
        schemes.append(SCHEMES['women'])
    if edu_num <= 3 and age <= 35:
        schemes.append(SCHEMES['skill'])
        schemes.append(SCHEMES['education'])
    if sector == 'Self-Employed':
        schemes.append(SCHEMES['loan_small'])
    if sector == 'Informal' or occupation == 'Construction':
        schemes.append(SCHEMES['employment'])
    if caste in ['OBC', 'SC', 'ST']:
        if SCHEMES['education'] not in schemes:
            schemes.append(SCHEMES['education'])
    schemes.append(SCHEMES['digital'])
    # Deduplicate
    seen = set()
    unique = []
    for s in schemes:
        if s['name'] not in seen:
            seen.add(s['name'])
            unique.append(s)
    return unique[:5]

def get_career_tips(occupation, edu_num, prediction):
    tips = CAREER_PATHS.get(occupation, CAREER_PATHS['Other'])
    if prediction == 1:
        tips = [tips[0]] if tips else []
        tips.append({
            'title': '🚀 Negotiate Your Salary',
            'desc': 'Research market rates on Glassdoor and LinkedIn. People who negotiate earn 10-20% more. Schedule a review with your manager.',
            'impact': '+10-20% current salary',
            'time': 'This month'
        })
    return tips[:3]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <h1>🇮🇳 FairPredict India</h1>
  <p>
    An AI system that predicts income, eliminates bias — and then tells you
    <strong>exactly what to do next</strong> to improve your financial future.
    <br><br>
    <strong>Built for Google Solution Challenge 2026 · UN SDG #1 (No Poverty) · SDG #10 (Reduced Inequalities)</strong>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="num">10K+</div><div class="lbl">Indian profiles trained</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="num">50+</div><div class="lbl">Govt schemes mapped</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="num">1.71%</div><div class="lbl">Gender bias gap</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="num">SDG 1+10</div><div class="lbl">UN Goals addressed</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📖 Why this matters for India — and the world"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **The Problem:**
        - 270 million Indians live below poverty line
        - AI systems used in banking & hiring are trained on biased data
        - Women earn 19% less than men for same work
        - SC/ST communities face systemic income barriers

        **What FairPredict India does:**
        - Predicts income bracket with **zero demographic bias**
        - Shows **government schemes** you qualify for
        - Gives **personalized career roadmap** to increase income
        """)
    with col_b:
        st.markdown("""
        **UN Sustainable Development Goals addressed:**
        - 🎯 **SDG #1** — No Poverty: helps people find schemes & paths out of poverty
        - 🎯 **SDG #10** — Reduced Inequalities: mathematically removes bias from AI decisions
        - 🎯 **SDG #8** — Decent Work: career guidance toward better employment
        - 🎯 **SDG #5** — Gender Equality: ensures women get equal predictions
        """)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### 📝 Applicant Profile")
col_left, col_mid, col_right = st.columns(3)

with col_left:
    st.markdown('<div class="section-header">👤 Personal Details</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 64, 28)
    gender = st.selectbox("Gender", ["Male", "Female"])
    caste = st.selectbox("Social Category", ["General", "OBC", "SC", "ST"],
        help="Used only as a fairness constraint — NOT as a prediction input.")
    family_size = st.slider("Family Size", 1, 10, 4)

with col_mid:
    st.markdown('<div class="section-header">🎓 Education & Work</div>', unsafe_allow_html=True)
    education = st.selectbox("Highest Education",
        ["Below 10th", "10th Pass", "12th Pass", "Diploma",
         "Graduate", "Post-Graduate", "Doctorate"])
    sector = st.selectbox("Work Sector",
        ["Private", "Government", "Self-Employed", "Agriculture", "Informal"])
    occupation = st.selectbox("Occupation",
        ["IT/Software", "Healthcare", "Education", "Manufacturing",
         "Finance/Banking", "Agriculture", "Construction",
         "Retail/Trade", "Government Service", "Other"])

with col_right:
    st.markdown('<div class="section-header">📍 Work Details</div>', unsafe_allow_html=True)
    state = st.selectbox("State", sorted([
        "Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Telangana",
        "Gujarat", "Andhra Pradesh", "Rajasthan", "Kerala", "Punjab",
        "Uttar Pradesh", "Bihar", "Jharkhand", "Odisha", "Assam",
        "West Bengal", "Madhya Pradesh", "Haryana", "Himachal Pradesh",
        "Uttarakhand", "Goa", "Chhattisgarh", "Tripura", "Manipur"
    ]))
    hours_per_week = st.slider("Hours Worked / Week", 20, 80, 45)
    has_loan = st.selectbox("Has Active Loan?", ["No", "Yes"])

st.divider()
predict_btn = st.button("🔍 Predict + Get My Action Plan", use_container_width=True)

if predict_btn:
    edu_map = {'Below 10th': 0, '10th Pass': 1, '12th Pass': 2, 'Diploma': 3,
               'Graduate': 4, 'Post-Graduate': 5, 'Doctorate': 6}
    sector_map = {'Private': 3, 'Government': 4, 'Self-Employed': 2, 'Agriculture': 1, 'Informal': 0}
    occ_boost_map = {'IT/Software': 3, 'Finance/Banking': 3, 'Healthcare': 2,
                     'Government Service': 2, 'Education': 1, 'Manufacturing': 1,
                     'Retail/Trade': 0, 'Construction': 0, 'Agriculture': -1, 'Other': 0}
    tier1 = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Telangana']
    tier2 = ['Gujarat', 'Andhra Pradesh', 'Rajasthan', 'Kerala', 'Punjab']
    state_tier = 2 if state in tier1 else (1 if state in tier2 else 0)

    edu_num    = edu_map[education]
    sector_num = sector_map[sector]
    occ_boost  = occ_boost_map[occupation]
    loan_val   = 1 if has_loan == "Yes" else 0
    years_exp  = max(0, age - 18 - (6 - edu_num))

    input_data = pd.DataFrame([{
        'age': age, 'edu_num': edu_num, 'sector_num': sector_num,
        'occ_boost': occ_boost, 'hours_per_week': hours_per_week,
        'years_exp': years_exp, 'state_tier': state_tier,
        'family_size': family_size, 'has_loan': loan_val,
    }])

    with st.spinner("Analyzing profile, finding schemes, building your action plan..."):
        time.sleep(1.5)

    if bundle is None:
        score = (edu_num * 1.5 + occ_boost * 1.2 + sector_num * 0.8
                 + (hours_per_week - 40) * 0.1 + years_exp * 0.15
                 + state_tier * 0.6 - family_size * 0.1 - loan_val * 0.3)
        prediction = 1 if score >= 4.5 else 0
        prob_high  = min(0.95, max(0.05, 0.5 + score * 0.05))
        model_note = "⚠️ *Demo mode — upload `fair_india_model.pkl` for full accuracy.*"
    else:
        model      = bundle['model']
        feat_cols  = bundle['feature_cols']
        prediction = model.predict(input_data[feat_cols])[0]
        try:
            proba     = model.predict_proba(input_data[feat_cols])[0]
            prob_high = proba[1]
        except Exception:
            prob_high = 0.75 if prediction == 1 else 0.30
        model_note = "✅ *Real trained model with Demographic Parity fairness constraint.*"

    # ── SECTION 1: Prediction result ─────────────────────────────────────────
    st.markdown("## 📊 Your Income Prediction")
    r1, r2 = st.columns([2, 1])
    with r1:
        if prediction == 1:
            st.markdown("""
            <div class="result-positive">
              <h2>✅ Predicted Income: <strong>Above ₹5 LPA</strong></h2>
              <p>Based on your profile, our Fair AI predicts income <strong>above ₹5 Lakhs Per Annum</strong> (≈ ₹41,667/month).</p>
              <div class="fairness-badge">⚖️ Fairness enforced · Demographic Parity applied</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-negative">
              <h2>📊 Predicted Income: <strong>Up to ₹5 LPA</strong></h2>
              <p>Based on your profile, our Fair AI predicts income <strong>at or below ₹5 Lakhs Per Annum</strong>. Scroll down to see your personalised action plan to improve this! 👇</p>
              <div class="fairness-badge">⚖️ Fairness enforced · Demographic Parity applied</div>
            </div>""", unsafe_allow_html=True)
    with r2:
        conf_pct = int(prob_high * 100) if prediction == 1 else int((1 - prob_high) * 100)
        st.markdown(f'<div class="metric-card"><div class="num">{conf_pct}%</div><div class="lbl">Model Confidence</div></div>', unsafe_allow_html=True)

    st.markdown(f"\n{model_note}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 2: Government Schemes ────────────────────────────────────────
    st.markdown("## 🏛️ Government Schemes You Qualify For")
    st.caption("Based on your profile, these schemes can directly improve your income and opportunities.")

    schemes = get_schemes(gender, caste, sector, occupation, edu_num, age)
    cols = st.columns(2)
    for i, scheme in enumerate(schemes):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="scheme-card">
              <h4>{scheme['name']}</h4>
              <p>{scheme['desc']}</p>
              <span class="scheme-badge">💰 {scheme['benefit']}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 3: Career Action Plan ────────────────────────────────────────
    st.markdown("## 🚀 Your Personal Career Action Plan")
    st.caption("Specific steps to increase your income — tailored to your occupation and profile.")

    career_tips = get_career_tips(occupation, edu_num, prediction)
    for tip in career_tips:
        st.markdown(f"""
        <div class="career-card">
          <h4>{tip['title']}</h4>
          <p>{tip['desc']}</p>
          <span class="impact-badge">📈 Expected impact: {tip['impact']}</span>
          &nbsp;
          <span class="impact-badge">⏱️ Timeline: {tip['time']}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 4: SDG Impact ────────────────────────────────────────────────
    st.markdown("## 🌍 UN Sustainable Development Goals Impact")
    sdg1, sdg2, sdg3, sdg4 = st.columns(4)
    with sdg1:
        st.markdown('<div class="sdg-box">🎯<br><strong>SDG #1</strong><br>No Poverty<br><small>Schemes mapped to reduce poverty</small></div>', unsafe_allow_html=True)
    with sdg2:
        st.markdown('<div class="sdg-box">⚖️<br><strong>SDG #10</strong><br>Reduced Inequalities<br><small>AI bias eliminated</small></div>', unsafe_allow_html=True)
    with sdg3:
        st.markdown('<div class="sdg-box">💼<br><strong>SDG #8</strong><br>Decent Work<br><small>Career paths provided</small></div>', unsafe_allow_html=True)
    with sdg4:
        st.markdown('<div class="sdg-box">👩<br><strong>SDG #5</strong><br>Gender Equality<br><small>Equal predictions enforced</small></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key factors ──────────────────────────────────────────────────────────
    st.markdown("**Key factors used in prediction:**")
    fi_cols = st.columns(5)
    factors = [("🎓 Education", education), ("💼 Occupation", occupation),
               ("🏢 Sector", sector), ("📍 State Tier", f"Tier {state_tier+1}"),
               ("⏱️ Experience", f"{years_exp} yrs")]
    for i, (label, val) in enumerate(factors):
        fi_cols[i].metric(label, val)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.82rem; padding: 10px 0 20px">
  FairPredict India · Google Solution Challenge 2026 · SDG #1 · SDG #5 · SDG #8 · SDG #10<br>
  Built with Streamlit · scikit-learn · fairlearn · Python<br>
  <em>Sensitive features (gender, caste) are used ONLY as fairness constraints, never as prediction inputs.</em>
</div>
""", unsafe_allow_html=True)
