# Customer Churn Prediction & Retention Strategy

Most churn prediction projects stop at the model. They output a probability score, show an AUC-ROC curve and call it done. The business question — which customers should we actually spend money on, and how much, and what should we do differently based on their risk level — goes unanswered.

I built this project to answer both halves. The model was the starting point, not the deliverable.

---

## What I set out to do

I worked with 5,630 customer records from an e-commerce platform and built a full pipeline from raw data through to a retention investment plan with ROI numbers that could go directly to a leadership meeting. Four classification models trained and compared, the best one selected, every customer scored and grouped into risk tiers, and then a financial model that showed exactly how much revenue was at risk, what the intervention cost and what the expected return was.

The financial model at the end was the part I spent the most time on. Predicting churn is a solved problem at this point. Turning that into "spend $23,357 on these specific customers using these specific interventions and expect to save $503,848 in annual revenue" — that is what stakeholders can actually act on.

---

## Results

**Model performance**

| Model | AUC-ROC | F1 Score | CV AUC (5-fold) |
|:---|:---|:---|:---|
| Logistic Regression | **0.8072** | 0.5585 | 0.8178 ± 0.004 |
| XGBoost | 0.7817 | 0.5240 | 0.7944 ± 0.015 |
| Random Forest | 0.7783 | 0.5191 | 0.7905 ± 0.014 |
| Decision Tree | 0.7516 | 0.5241 | 0.7244 ± 0.023 |

Logistic Regression won at 0.8072 AUC-ROC. What made it the right choice beyond the headline number was the 5-fold cross-validation variance — ±0.004 was the tightest of the four models, meaning it generalised well rather than fitting the training data closely and performing worse on new customers. XGBoost had a higher peak on some runs but a much wider variance band.

**Customer risk breakdown**

| Risk Tier | Customers | Actual Churn Rate | Avg Monthly Revenue |
|:---|:---|:---|:---|
| High Risk (prob ≥ 70%) | 1,147 (20.4%) | 60.2% | $68.58 |
| Medium Risk (prob 30–70%) | 2,259 (40.1%) | 25.5% | $65.88 |
| Low Risk (prob < 30%) | 2,224 (39.5%) | 5.6% | $63.90 |

**The financial case**

```
Annual revenue at risk (High + Medium Risk customers)   →  $2,729,704
Total retention intervention cost                       →     $23,357
Expected annual revenue saved (conservative estimates)  →    $503,848
Net benefit                                             →    $480,491
ROI                                                     →      2,057%

Every $1 spent on retention recovered $21.57 in revenue.
```

---

## What actually drove churn

These came from the Random Forest feature importance scores. I found each one was actionable, which was the criterion I used to decide whether to include it in the business section:

| Feature | Importance | What the data suggested |
|:---|:---|:---|
| Tenure | 0.1805 | New customers (0–6 months) churned at the highest rate — the onboarding period was the biggest retention risk |
| CLV Proxy | 0.1390 | Low-lifetime-value customers churned more — the model was partly picking up acquisition quality |
| Satisfaction Score | 0.1194 | Low scorers churned heavily — satisfaction recovery needed to be immediate, not at the next quarterly review |
| Complaints | 0.0866 | Filing a complaint was a major risk signal regardless of how it was resolved |
| Cashback Amount | 0.0550 | Lower cashback engagement correlated with lower stickiness to the platform |
| Days Since Last Order | 0.0499 | A customer going quiet was an early warning sign even before satisfaction scores dropped |

The complaints finding was the one worth flagging separately. Customers who filed a complaint and received a low satisfaction score were the highest immediate churn risk in the dataset — an action that could be taken without waiting for any model output, by any CRM team that already tracked complaint resolution.

---

## The dataset

- 5,630 customers across 21 features
- 24.7% churn rate (1,393 churned customers)
- E-commerce platform — customers across East Africa, Middle East, Europe and North America
- Behavioural signals, order history, satisfaction scores, complaint data, demographics

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

## Feature engineering

I added five features that were not in the raw data but carried meaningful signal when I looked at how churn rates split across different customer segments:

```python
# Customer Lifetime Value proxy — simple but effective
df['CLV_Proxy'] = df['Tenure'] * df['MonthlyCharges']

# Composite engagement score
df['EngagementScore'] = (
    df['HourSpendOnApp'] * 2 +
    df['OrderCount'] * 1.5 +
    df['CouponUsed'] * 0.5 -
    df['DaySinceLastOrder'] * 0.3
)

# Tenure buckets — churn rates differed significantly between groups
# New (0-6m):       1,744 customers — highest churn risk
# Growing (7-18m):  2,050 customers
# Established (19-36m): 1,173 customers
# Loyal (36m+):       663 customers — lowest churn risk

# Rule-based high risk flag — fast pre-screen for CRM teams
df['HighRiskFlag'] = (
    (df['SatisfactionScore'] <= 2) |
    (df['Complain'] == 1) |
    (df['DaySinceLastOrder'] >= 10)
).astype(int)
```

The CLV_Proxy ended up being the second most important feature at 0.139 importance. It is a simple calculation — tenure multiplied by monthly charges — but it captured something important. A customer with 3 months tenure paying $100 per month looked very different from one with 36 months at $30 per month, even if their current behaviour looked similar in the raw features.

---

## The threshold decision

The default 0.5 classification threshold was not the right choice for this problem and I did not use it.

The asymmetry in costs made this clear. Missing a customer who was going to churn — a false negative — cost the business roughly $50 in lost annual revenue per customer. Spending on retention for someone who was not going to leave — a false positive — cost around $5 in campaign spend. With that cost ratio, the model should be set to catch more true positives even at the cost of more false positives.

I modelled this explicitly and found that the business-cost-minimising threshold was **0.21**, not 0.50. This is what I used for the risk segmentation and retention plan. It captured significantly more true positives and the math supported the trade-off.

---

## Retention plan

The scored customer list was the input to a retention investment plan, not the final output.

| Risk Tier | Revenue Profile | Intervention | Cost per Customer |
|:---|:---|:---|:---|
| High Risk + High Revenue (≥$65/mo) | 617 customers | Personal outreach + loyalty discount | $20 |
| High Risk + Low Revenue (<$65/mo) | 530 customers | Automated email + cashback offer | $8 |
| Medium Risk | 2,259 customers | Monthly engagement campaign | $3 |
| Low Risk — New Customers | 210 customers | Onboarding programme | $1 |
| Low Risk — Established | 2,014 customers | Newsletter + recommendations | $1 |

The split between High Risk + High Revenue and High Risk + Low Revenue was a deliberate design choice. Spending $20 on personal outreach for a customer generating $89 per month made clear financial sense. Spending $20 on someone paying $40 per month did not. The model gave the targeting; the intervention design reflected the expected return from each segment.

---

## Project structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction_Model.ipynb      # Main notebook — 15 sections
│
├── data/
│   ├── raw/
│   │   └── ecommerce_churn.csv                # Raw dataset (5,630 records)
│   └── processed/
│       ├── ecommerce_churn_clean.csv          # Cleaned + engineered features
│       ├── ecommerce_churn_powerbi.csv        # Includes ChurnProbability + RiskTier
│       └── model_performance_summary.csv      # All 4 model metrics
│
├── models/
│   ├── best_churn_model.pkl                   # Trained Logistic Regression + scaler
│   └── churn_model.pkl                        # Production-ready model artifact
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
git clone https://github.com/PatienceAnono/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction_Model.ipynb
```

The notebook runs top to bottom without any manual steps between sections. Output directories for visuals, processed data and models are created automatically on first run.

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

To use the trained model directly without rerunning the full notebook:

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
# X_new should be a DataFrame with the same columns as features
# Encode categoricals with le_dict, then scale with scaler:
churn_proba = model.predict_proba(X_new_scaled)[:, 1]
```

The bundle contained everything needed to reproduce the preprocessing pipeline — label encoders for each categorical column, the StandardScaler fitted on training data, and the feature list in the correct order. Nothing needed to be retrained or manually reconstructed.

---

## Notebook walkthrough — 15 sections

1. **Executive Summary** — the business question and headline results, written before the technical sections so a stakeholder could read section 1 and understand the project without reading the rest
2. **Setup** — libraries, colour palette, output directory creation
3. **Data Loading & Cleaning** — quality audit, missing value treatment, validation checks
4. **EDA** — churn rates by category, numeric feature distributions, complaint analysis
5. **Feature Engineering** — CLV proxy, tenure groups, engagement score, high risk flag
6. **Preprocessing & Training** — 80/20 stratified split, label encoding, StandardScaler, all four models trained
7. **Model Evaluation** — ROC curves, confusion matrices, classification reports side by side
8. **Feature Importance** — Random Forest and XGBoost importance scores with business interpretation
9. **Threshold Optimisation** — F1-optimal vs business-cost-optimal threshold comparison with explicit cost modelling
10. **Risk Segmentation** — all 5,630 customers scored and assigned to High/Medium/Low tiers
11. **Retention Plan** — intervention type and cost per customer based on risk tier and revenue level
12. **ROI Analysis** — revenue at risk, expected savings, net benefit, full financial model
13. **Executive Dashboard** — single-page summary combining all key metrics into one chart
14. **Recommendations** — immediate actions, 90-day roadmap, model maintenance schedule
15. **Export** — cleaned CSV, Power BI dataset, model performance summary, saved model

---

## A note on the ROI numbers

The 2,057% ROI used conservative assumptions — 25% retention success rate for High Risk customers and 15% for Medium Risk. These were deliberately conservative. If the retention team executed well, the numbers would be higher. If campaign execution was weak, lower.

What the numbers were not sensitive to was the direction of the finding. $23,357 in intervention spend against $2.7 million in annual revenue at risk was a ratio that held up even with significantly more pessimistic assumptions about retention success rates. The cost of not acting was the more important number in that calculation.

---

*Dataset is synthetic, built for analytical demonstration. The modelling methodology, threshold analysis, feature engineering and business logic are real.*

---

**Patience Anono** · PA Data Analytics · [padataanalytics.com](https://padataanalytics.com) · hello@padataanalytics.com
