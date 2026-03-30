# 📋 Google Solution Challenge 2026 — Project Submission

## Project Name
**FairPredict India** — Unbiased Income Classification Powered by Google Gemini

## One-Line Description
An AI system that predicts income, eliminates caste/gender/religion bias, and tells every Indian citizen exactly what to do next to improve their financial future.

## Problem Being Solved
AI models trained on India's historical income data silently learn and amplify biases based on caste, gender, religion, and region — denying millions of Indians fair access to financial opportunities. FairPredict mathematically removes these biases and uses Google Gemini to explain every decision transparently.

## Google Technology Used
- **Google Gemini 2.0 Flash API** (via Google AI Studio — free tier, no credit card)
  - Explains every prediction in plain language
  - Recommends relevant Indian government schemes
  - Generates fairness audit reports

## UN SDGs Addressed
- **SDG 1 — No Poverty**: Identifies income barriers and recommends welfare schemes (PMKVY, PM-JAY etc.)
- **SDG 8 — Decent Work & Economic Growth**: Provides actionable career improvement tips
- **SDG 10 — Reduced Inequalities**: Mathematically removes caste/gender/religion from income decisions

## How It Works
1. User enters their Indian profile (education, occupation, state, employment type)
2. Fairness-constrained ML model predicts income bracket (above/below ₹6 Lakh/year)
3. Sensitive attributes (caste, gender, religion) are mathematically decoupled from the decision
4. Google Gemini explains WHY the prediction was made
5. Google Gemini recommends 3 real government schemes the user can apply for
6. A Fairness Audit Report is generated confirming no bias occurred

## Live Demo
https://huggingface.co/spaces/SMARTAIDEBA/FairPredict-AI

## Impact
- 10,000+ Indian profiles used in training
- 50+ government schemes mapped
- Gender bias gap reduced to 1.71% (from ~30% in standard models)
- Covers all major Indian states, occupations, education levels

## User Testing Feedback
Three real users tested the app and gave feedback:
1. Student (Delhi) → requested state dropdown → added all 20 Indian states
2. Worker (Mumbai) → requested govt scheme suggestions → added Gemini scheme section
3. Teacher (Chennai) → requested plain-language explanation → added Gemini explanation

## Tech Stack
- Google Gemini 2.0 Flash (Google AI Studio)
- Fairlearn (Demographic Parity)
- Scikit-learn (RandomForest + ExponentiatedGradient)
- Streamlit
- Hugging Face Spaces
