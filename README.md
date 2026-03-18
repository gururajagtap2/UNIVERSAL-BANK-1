# 🏦 Universal Bank — Personal Loan Intelligence Dashboard

A multi-tab Streamlit dashboard that walks through the full analytics journey — from raw data exploration to prescriptive marketing recommendations — powered by three ML classification models.

---

## 📂 Project Structure

```
universal_bank_dashboard/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── UniversalBank.csv       ← Training dataset (5,000 customers)
├── sample_test_data.csv    ← Sample test file (500 rows, no target column)
└── README.md               ← This file
```

---

## 🚀 Running Locally

```bash
# 1. Clone / unzip the project
cd universal_bank_dashboard

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud (streamlit.io)

1. Push this folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **"New app"**, select your repo, branch, and set the main file to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs `requirements.txt` automatically.

> **Important:** Make sure `UniversalBank.csv` is committed to the repo root alongside `app.py`.

---

## 📊 Dashboard Tabs

| Tab | Analytics Type | What It Shows |
|-----|---------------|---------------|
| 📋 Overview | Descriptive | Dataset preview, data quality checks, class distribution |
| 📊 Descriptive Analytics | Descriptive | Feature distributions, correlation heatmap |
| 🔍 Diagnostic Analytics | Diagnostic | Acceptance rates by segment, income/CCAvg scatter |
| 🤖 Predictive Models | Predictive | Metrics table, ROC curve, confusion matrices |
| 🎯 Prescriptive Insights | Prescriptive | Feature importance, ideal customer profile, campaign matrix |
| 📤 Predict New Data | Applied | Upload CSV → predict → download results |

---

## 🤖 Models

| Model | Class Imbalance Handling |
|-------|--------------------------|
| Decision Tree | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"` |
| Gradient Boosted Tree | Minority class weighted via sample distribution |

**Train / Test split:** 70 % / 30 % stratified by target class.

---

## 📤 Using the Prediction Feature

Upload `sample_test_data.csv` (included) to the **"Predict New Data"** tab to test the prediction pipeline. The app will:

1. Preprocess the uploaded file (fix negative Experience, drop ID/ZIP)
2. Score each customer using the best model (by ROC-AUC)
3. Add columns: `Predicted_Personal_Loan`, `Loan_Probability_%`, `Priority_Tier`
4. Let you download the enriched file as CSV

---

## 📋 Column Reference

| Column | Description |
|--------|-------------|
| ID | Customer identifier (dropped before modelling) |
| Age | Age in years |
| Experience | Years of professional experience |
| Income | Annual income ($000) |
| ZIP Code | Home ZIP code (dropped before modelling) |
| Family | Family size (1–4) |
| CCAvg | Average monthly credit card spend ($000) |
| Education | 1 = Undergrad, 2 = Graduate, 3 = Advanced/Professional |
| Mortgage | Mortgage value ($000) |
| Personal Loan | **Target** — 1 = accepted loan, 0 = rejected |
| Securities Account | 1 = holds securities account |
| CD Account | 1 = holds certificate of deposit |
| Online | 1 = uses internet banking |
| CreditCard | 1 = holds bank-issued credit card |
