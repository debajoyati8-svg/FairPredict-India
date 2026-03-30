# 🇮🇳 FairPredict India
### Unbiased Income Classification — Powered by Google Gemini AI

[![Google Solution Challenge 2026](https://img.shields.io/badge/Google%20Solution%20Challenge-2026-4285F4?style=for-the-badge&logo=google)](https://developers.google.com/community/gdsc-solution-challenge)
[![Powered by Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-EA4335?style=for-the-badge&logo=google)](https://aistudio.google.com)
[![Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn%20Demographic%20Parity-34A853?style=for-the-badge)](https://fairlearn.org)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-FBBC04?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/SMARTAIDEBA/FairPredict-AI)

---

## 🎯 Problem Statement

> *"Computer programs now make life-changing decisions about who gets a job, a bank loan, or even medical care. However, if these programs learn from flawed or unfair historical data, they will repeat and amplify those exact same discriminatory mistakes."*
> — Google Solution Challenge 2026 Problem Track: **Unbiased AI Decision**

In India, wage inequality is deeply linked to **caste, gender, religion, and region** — not just skills and education. When AI models are trained on historical income data, they silently learn and amplify these biases, creating a vicious cycle of discrimination at scale.

---

## 💡 Our Solution

**FairPredict India** is an AI-powered income prediction system that:

- 🔒 **Mathematically removes bias** using Fairlearn's Demographic Parity constraint
- 🤖 **Explains every decision** in plain language using **Google Gemini AI**
- 🏛️ **Recommends real Indian government schemes** tailored to each applicant
- 📋 **Generates a Fairness Audit Report** for every prediction
- 🇮🇳 **Built for India** — with Indian occupations, education levels, states, and ₹ income labels

---

## 🏗️ Architecture

```
User Input (Indian Profile)
        ↓
Fairlearn Model (Demographic Parity)
  → Sensitive attributes (Caste/Gender/Religion) DECOUPLED
        ↓
Income Prediction (₹ bracket)
        ↓
Google Gemini API ──→ Plain-language Explanation
                  ──→ Govt Scheme Recommendations  
                  ──→ Fairness Audit Report
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| 🤖 AI Explanation | **Google Gemini 2.0 Flash** (Google AI Studio) |
| ⚖️ Fairness | **Fairlearn** — Demographic Parity |
| 🧠 ML Model | **Scikit-learn** RandomForest + ExponentiatedGradient |
| 🖥️ Frontend | **Streamlit** |
| ☁️ Hosting | **Hugging Face Spaces** |

---

## 🎯 UN SDG Alignment

| SDG | How FairPredict Addresses It |
|-----|------------------------------|
| 🟠 **SDG 1** — No Poverty | Identifies income barriers and recommends welfare schemes |
| 🟢 **SDG 8** — Decent Work & Economic Growth | Provides actionable career improvement tips |
| 🔵 **SDG 10** — Reduced Inequalities | Removes caste/gender/religion bias from income decisions |

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Training profiles | 10,000+ Indian profiles |
| Government schemes mapped | 50+ |
| Gender bias gap (after fairness) | 1.71% |
| UN SDGs addressed | 3 (SDG 1, 8, 10) |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/FairPredict-India.git
cd FairPredict-India

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key (free from aistudio.google.com)
export GEMINI_API_KEY="your_key_here"

# 4. Run the app
streamlit run app.py
```

---

## 🔑 Getting a Free Gemini API Key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with any Gmail account
3. Click **"Get API Key"** → **"Create API Key"**
4. Copy and set as environment variable — **no credit card needed!**

---

## 📁 Project Structure

```
FairPredict-India/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training with fairness constraints
├── fair_india_model.pkl    # Trained fairness-constrained model
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚖️ How Fairness Works

Traditional ML models trained on biased data learn to associate caste, gender, or religion with income — and then discriminate. FairPredict uses **Demographic Parity** to mathematically enforce equal positive prediction rates across all demographic groups.

```
Without FairPredict:  Male → 62% chance of >₹6L  |  Female → 31% chance
With FairPredict:     Male → 48% chance of >₹6L  |  Female → 46% chance ✅
```

---

## 👥 User Feedback (Tested With Real Users)

| User | Feedback | Change Made |
|------|----------|-------------|
| Student, Delhi | "Add my state to the dropdown" | Added all 20 Indian states ✅ |
| Worker, Mumbai | "Show what schemes I qualify for" | Added Gemini Govt Scheme section ✅ |
| Teacher, Chennai | "Explain why in simple language" | Added Gemini plain-language explanation ✅ |

---

## 🏆 Built For

**Google Solution Challenge 2026 — Build with AI**
Track: *[Unbiased AI Decision] Ensuring Fairness and Detecting Bias in Automated Decisions*

---

## 📄 License

MIT License — Free to use, modify, and distribute.
