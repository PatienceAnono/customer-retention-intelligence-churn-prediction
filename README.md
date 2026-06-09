# Customer Churn Prediction & Retention Strategy

This project started with a simple question a lot of e-commerce businesses avoid asking directly: **which of our customers are about to leave, and what should we actually do about it?**

I built an end-to-end machine learning solution on 5,630 customer records that answers both halves of that question. Not just a model that outputs churn probabilities and calls it done — but a full pipeline from raw data through to a retention investment plan with ROI numbers you can take to a leadership meeting.

---

## What this project does

Takes customer behavioural data, trains four classification models, picks the best one, scores every customer with a churn probability, groups them into risk tiers, and then builds a retention campaign plan that shows how much revenue you're likely to save versus what the intervention costs.

The financial model at the end is the part most ML portfolios skip. Predicting churn is straightforward. Turning that into a business case — "spend $23,357 on these specific interventions and expect to save $503,848 in annual revenue" — is what actually gets bought by a stakeholder.

---

## Results

**Model Performance**

| Model | AUC-ROC | F1 Score | CV AUC (5-fold) |
|-------|---------|----------|-----------------|
| Logistic Regression | **0.8072** | 0.5585 | 0.8178 ± 0.004 |
| XGBoost | 0.7817 | 0.5240 | 0.7944 ± 0.015 |
| Random Forest | 0.7783 | 0.5191 | 0.7905 ± 0.014 |
| Decision Tree | 0.7516 | 0.5241 | 0.7244 ± 0.023 |

Logistic Regression came out on top at 0.8072 AUC-ROC — which puts it firmly in the "very good" range for churn prediction. The 5-fold cross-validation also showed the tightest variance (±0.004), meaning the model generalises well rather than just fitting the training data.

**Customer Risk Breakdown**

| Risk Tier | Customers | Actual Churn Rate | Avg Monthly Revenue |
|-----------|-----------|-------------------|---------------------|
| High Risk (prob ≥ 70%) | 1,147 (20.4%) | 60.2% | $68.58 |
| Medium Risk (prob 30–70%) | 2,259 (40.1%) | 25.5% | $65.88 |
| Low Risk (prob < 30%) | 2,224 (39.5%) | 5.6% | $63.90 |

**The Financial Case**

```
Annual revenue at risk (High + Medium Risk customers)   →   $2,729,704
Total retention intervention cost                       →      $23,357
Expected annual revenue saved (conservative estimates)  →     $503,848
Net benefit                                             →     $480,491
ROI                                                     →       2,057%

Every $1 spent on retention recovers $21.57 in revenue.
```

---

## What actually drives churn

These are the top features from Random Forest, and they're worth paying attention to because each one is actionable:

| Feature | Importance | What to do about it |
|---------|-----------|---------------------|
| Tenure | 0.1805 | New customers (0–6 months) churn most — fix your onboarding |
| CLV Proxy | 0.1390 | Low-lifetime-value customers churn more — adjust targeting |
| Satisfaction Score | 0.1194 | Low scorers churn heavily — satisfaction recovery should be immediate |
| Complaints | 0.0866 | Filed a complaint = major risk signal — fix resolution speed |
| Cashback Amount | 0.0550 | Less cashback engagement = lower stickiness |
| Days Since Last Order | 0.0499 | Customer going quiet = early warning sign |

The interesting one is complaints. Customers who filed a complaint and got a low satisfaction score are your highest immediate churn risk — that's a retention action you can take *today* without waiting for any model output.

---

## Dataset

- **5,630 customers** across 21 features
- **24.7% churn rate** (1,393 churned customers)
- E-commerce platform — East Africa / Middle East / Europe / North America
- Regions, demographics, behavioural signals, order history, complaint data

**Columns used in training (16 features after engineering):**

```
Behavioural:    Tenure, HourSpendOnApp, DaySinceLastOrder, OrderCount, CouponUsed
Financial:      MonthlyCharges, CashbackAmount, OrderAmountHikeFromLastYear
Satisfaction:   SatisfactionScore, Complain
Demographic:    CityTier, WarehouseToHome, NumberOfDeviceRegistered, NumberOfAddress
Categorical:    PreferredLoginDevice, PreferredPaymentMode, Gender, MaritalStatus, PreferedOrderCat
Engineered:     CLV_Proxy (Tenure × MonthlyCharges), EngagementScore (composite)
```

---

## Engineered Features

I added five features that don't exist in the raw data but carry meaningful signal:

```python
# Customer Lifetime Value proxy
df['CLV_Proxy'] = df['Tenure'] * df['MonthlyCharges']

# Composite engagement score
df['EngagementScore'] = (
    df['HourSpendOnApp'] * 2 +
    df['OrderCount'] * 1.5 +
    df['CouponUsed'] * 0.5 -
    df['DaySinceLastOrder'] * 0.3
)

# Tenure buckets (churn rates differ significantly between groups)
# New (0-6m): 1,744 customers — highest churn risk
# Growing (7-18m): 2,050 customers
# Established (19-36m): 1,173 customers
# Loyal (36m+): 663 customers — lowest churn risk

# Rule-based high risk flag (fast pre-screen for CRM teams)
df['HighRiskFlag'] = (
    (df['SatisfactionScore'] <= 2) |
    (df['Complain'] == 1) |
    (df['DaySinceLastOrder'] >= 10)
).astype(int)
```

The CLV_Proxy ended up being the second most important feature (0.139 importance score). Simple calculation but it captures a lot — a customer with 3 months tenure paying $100/month looks very different from one with 36 months at $30/month, even if their current spend looks similar.

---

## Threshold Decision

The default 0.5 decision threshold is rarely the right choice for churn. If you're running a retention campaign, missing a churner is much more expensive than spending $8 on an email sequence for someone who wasn't going to leave anyway.

I modelled this explicitly:
- Cost of a false positive (retention spend on a non-churner): **$5**
- Cost of a false negative (missing an actual churner): **$50**

Business-cost-minimising threshold came out at **0.21**, which is what I used for the risk segmentation and retention plan. This captures more true positives at the cost of more false positives — but the math supports it given the cost asymmetry.

---

## Retention Plan

Rather than just dumping a scored CSV, I built out what you'd actually do with the predictions:

| Risk Tier | Revenue Profile | Intervention | Cost/Customer |
|-----------|----------------|-------------|---------------|
| High Risk + High Revenue (≥$65/mo) | 617 customers | Personal outreach + loyalty discount | $20 |
| High Risk + Low Revenue (<$65/mo) | 530 customers | Automated email + cashback offer | $8 |
| Medium Risk | 2,259 customers | Monthly engagement campaign | $3 |
| Low Risk — New Customers | 210 customers | Onboarding programme | $1 |
| Low Risk — Established | 2,014 customers | Newsletter + recommendations | $1 |

The budget split reflects the asymmetry in expected return. Spending $20 on a High Risk customer earning you $89/month is a completely different calculation than spending $20 on someone paying $40/month.

---

## Project Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction_Model.ipynb     ← Main notebook (15 sections)
│
├── data/
│   ├── raw/
│   │   └── ecommerce_churn.csv               ← Raw dataset (5,630 records)
│   └── processed/
│       ├── ecommerce_churn_clean.csv          ← Cleaned + engineered features
│       ├── ecommerce_churn_powerbi.csv        ← Includes ChurnProbability + RiskTier
│       └── model_performance_summary.csv      ← All 4 model metrics
│
├── models/
│   ├── best_churn_model.pkl                  ← Trained Logistic Regression + scaler
│   └── churn_model.pkl                       ← Production-ready model artifact
│
├── visuals/
│   ├── 1_churn_distribution.png
│   ├── 2_churn_by_category.png
│   ├── 3_numeric_vs_churn.png
│   ├── 4_complain_vs_churn.png
│   ├── 5_engineered_features.png
│   ├── 6_auc_roc.png
│   ├── 7_confusion_matrices.png
│   ├── 8_feature_importance.png
│   ├── 9_threshold_optimisation.png
│   ├── 10_risk_segmentation.png
│   ├── 11_intervention_plan.png
│   ├── 12_roi_analysis.png
│   └── 13_executive_dashboard.png
│
└── README.md
```

---

## How to run it

```bash
git clone https://github.com/anonopatience/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction_Model.ipynb
```

The notebook runs top to bottom with no manual steps. Directories for visuals, processed data and models are created automatically.

**Requirements:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## Loading the saved model

If you want to use the trained model directly without running the full notebook:

```python
import pickle
import pandas as pd

with open('models/best_churn_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model    = bundle['model']
scaler   = bundle['scaler']
features = bundle['features']
le_dict  = bundle['le_dict']

# Score new customers
# X_new = your dataframe with the same columns as `features`
# Encode categoricals with le_dict, scale with scaler, then:
churn_proba = model.predict_proba(X_new_scaled)[:, 1]
```

The bundle contains everything needed to reproduce the preprocessing pipeline — label encoders for each categorical column, the scaler, and the feature list. No need to retrain.

---

## Notebook walkthrough

The notebook is 15 sections — here's the short version of what each one does:

1. **Executive Summary** — the business question and headline results up front
2. **Setup** — libraries, colour palette, output directories
3. **Data Loading & Cleaning** — quality report, missing value treatment, validation checks
4. **EDA** — churn by category, numeric distributions, complaint analysis
5. **Feature Engineering** — CLV proxy, tenure groups, engagement score, high risk flag
6. **Preprocessing & Training** — 80/20 stratified split, label encoding, StandardScaler, train all 4 models
7. **Model Evaluation** — ROC curves, confusion matrices, classification reports
8. **Feature Importance** — Random Forest and XGBoost importance scores, business interpretation
9. **Threshold Optimisation** — F1-optimal vs business-cost-optimal threshold comparison
10. **Risk Segmentation** — score all 5,630 customers, assign High/Medium/Low tiers
11. **Retention Plan** — intervention type and cost per customer based on risk + revenue
12. **ROI Analysis** — revenue at risk, expected savings, net benefit, full financial model
13. **Executive Dashboard** — single-page summary combining all key metrics
14. **Recommendations** — immediate actions, short-term roadmap, model maintenance schedule
15. **Export** — clean CSV, Power BI dataset, model performance summary, model pkl

---

## A note on the ROI numbers

The 2,057% ROI figure uses conservative assumptions — 25% retention success for High Risk customers and 15% for Medium Risk. If your actual customer service and retention team is good, these numbers could be higher. If your campaign execution is weak, lower. The model gives you the targeting; the ROI depends on what you do with it.

The $23,357 intervention budget covers all 3,406 High + Medium Risk customers. That's not a big number relative to $2.7M in annual revenue at risk. The risk of *not* acting is significantly higher than the cost of acting.

---

## About

**Patience Anono** — Data Analyst & Analytics Consultant

I work with e-commerce brands and marketing teams on analytics projects that go all the way from raw data to business decisions. If you're sitting on customer data and not sure what it's telling you, feel free to reach out.

📧 hello@padataanalytics.com  
🌐 [padataanalytics.com](https://padataanalytics.com)  
💼 [LinkedIn](https://www.linkedin.com/in/patience-anono-22ab06176/)  
💻 [GitHub](https://github.com/anonopatience)

---

*Dataset is synthetic, built for analytical demonstration. The modelling methodology and business logic are real.*
