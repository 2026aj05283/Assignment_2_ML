# 📡 ChurnLens — Telecom Customer Churn Classification

**ML Assignment 2 · M.Tech (AIML), BITS Pilani WILP**
**Name:** Kunal Vaishnav  **BITS ID:** 2026aj05283

| | |
|---|---|
| **GitHub Repository** | https://github.com/<your-github-username>/ml-assignment-2 |
| **Live Streamlit App** | https://<your-app-name>.streamlit.app |

---

## a. Problem statement

A telecom operator loses revenue every time a subscriber cancels the service ("churns"). Acquiring a new customer costs several times more than retaining an existing one, so the business wants to **identify customers who are likely to churn before they leave** and target them with retention offers.

The task is framed as a **binary classification** problem: given a customer's account information and usage behaviour (call minutes, charges, plans, customer-service calls), predict whether the customer will churn (`churn = 1`) or stay (`churn = 0`).

Five classification algorithms — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine — are trained on the same data, evaluated with six metrics, compared, and deployed in an interactive Streamlit app.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Telecom Customer Churn (BigML "Churn in Telecom's dataset", Kaggle) — obtained via the PMLB benchmark repository (`churn`) |
| Source | Kaggle / BigML; mirrored by PMLB (https://github.com/EpistasisLab/pmlb) |
| Instances | **5,000** customers |
| Input features | **19** (after dropping the `phone_number` identifier) |
| Target | `churn` — 0 = stayed (4,293 · 85.9 %), 1 = churned (707 · 14.1 %) |
| Missing values | None |
| Task | Binary classification (imbalanced) |

**Features**

| Feature | Description |
|---|---|
| `state` | US state (label-encoded, 51 values) |
| `account_length` | Days the account has been active |
| `area_code` | Telephone area code (408 / 415 / 510) |
| `international_plan` | 1 if the customer has an international plan |
| `voice_mail_plan` | 1 if the customer has a voice-mail plan |
| `number_vmail_messages` | Number of voice-mail messages |
| `total_day_minutes`, `total_day_calls`, `total_day_charge` | Day-time usage |
| `total_eve_minutes`, `total_eve_calls`, `total_eve_charge` | Evening usage |
| `total_night_minutes`, `total_night_calls`, `total_night_charge` | Night usage |
| `total_intl_minutes`, `total_intl_calls`, `total_intl_charge` | International usage |
| `number_customer_service_calls` | Calls made to customer service |

**Interesting properties observed during EDA**
- Customers with an international plan churn at **42.1 %** vs 11.2 % without one.
- Customers with **≥ 4 customer-service calls** churn at **50.4 %** vs 11.0 % for the rest.
- Each `*_charge` column is an exact linear function of its `*_minutes` column (correlation ≈ 1.0), i.e. the feature set contains perfectly redundant pairs.

**Train / test protocol**
- 80 / 20 **stratified** split, `random_state = 2026` → 4,000 training rows, 1,000 test rows.
- The 1,000-row hold-out is saved as `test_data.csv` and was never used for training or tuning.
- Hyper-parameters tuned with **5-fold stratified GridSearchCV on the training set only**, optimising F1 (better suited than accuracy for the 14 % minority class).
- Each model is saved as a full scikit-learn `Pipeline` (StandardScaler + classifier; the Decision Tree needs no scaling), so the app applies exactly the same preprocessing.

## c. GitHub Repository Link

https://github.com/<your-github-username>/ml-assignment-2

```
ml-assignment-2/
│-- app.py                    # Streamlit app (loads saved models, never retrains)
│-- train_models.py           # Training + evaluation script for all 5 models
│-- requirements.txt          # Pinned dependencies for Streamlit Cloud
│-- runtime.txt               # Python version hint (3.12)
│-- README.md
│-- test_data.csv             # 1,000-row hold-out test set (features + churn)
│-- data/
│   └── telecom_churn.csv     # Full cleaned dataset (5,000 rows)
│-- model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metrics.json          # Test metrics + tuned hyper-parameters
│   └── feature_meta.json     # Feature order / target used by the app
└── .streamlit/config.toml    # Custom theme
```

## d. Models used

All five models were trained on the same 4,000 training rows and evaluated on the same 1,000-row test set (`test_data.csv`). Positive class = churn (1). AUC is computed from predicted probabilities.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7910 | 0.8306 | 0.3803 | 0.7660 | 0.5082 | 0.4330 |
| Decision Tree | **0.9510** | 0.8817 | **0.9035** | 0.7305 | **0.8078** | **0.7859** |
| KNN | 0.8940 | 0.8037 | 0.7397 | 0.3830 | 0.5047 | 0.4828 |
| Naive Bayes | 0.8730 | 0.8515 | 0.5538 | 0.5106 | 0.5314 | 0.4586 |
| Support Vector Machine | 0.9350 | **0.9048** | 0.8393 | 0.6667 | 0.7431 | 0.7126 |

**Tuned hyper-parameters**

| Model | Best parameters (5-fold CV, F1) |
|---|---|
| Logistic Regression | C = 0.1, class_weight = balanced |
| Decision Tree | criterion = entropy, max_depth = 8, min_samples_leaf = 10 |
| KNN | n_neighbors = 3, weights = uniform |
| Naive Bayes (Gaussian) | var_smoothing = 1e-9 |
| SVM (RBF kernel) | C = 10, gamma = scale, class_weight = None |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Weakest overall (MCC 0.433). Churn depends on non-linear threshold effects — e.g. risk jumps sharply after the 4th customer-service call and at very high day-time usage — which a single linear boundary cannot represent. Grid search chose `class_weight="balanced"`, giving the highest recall of all models (76.6 %) but very poor precision (38 %): 176 loyal customers were flagged as churners. Useful as an interpretable baseline or when missing a churner is far costlier than a false alarm. |
| Decision Tree | Best model on this dataset (Accuracy 0.951, F1 0.808, MCC 0.786). The data is naturally rule-shaped and the learned tree shows it: the root split is `total_day_charge > 41.93`, followed by `number_vmail_messages ≤ 6.5`, `number_customer_service_calls > 3.5` and `international_plan = 1 with total_intl_calls ≤ 2` — exactly the threshold effects a linear model misses. Top features: `total_day_charge` (31 %), `number_customer_service_calls` (15 %), `international_plan` (10 %). Depth limit 8 and min 10 samples per leaf kept it from overfitting. Only 11 false positives, though its AUC (0.882) is lower than SVM because its probability estimates are coarse (leaf proportions). |
| KNN | High accuracy (0.894) is misleading — recall is only 38 %, so it missed 87 of 141 churners. With 19 scaled dimensions, uninformative features such as `state`, `area_code` and the call counts dilute the distance metric, and with only 14 % positives most neighbourhoods are dominated by the majority class. Lowest AUC (0.804) of all models. |
| Naive Bayes | Middle of the pack (F1 0.531, MCC 0.459) with a reasonable AUC (0.852). Its independence assumption is clearly violated: every `*_charge` feature is a perfect linear copy of its `*_minutes` feature, so the same evidence is counted twice, pushing probabilities to extremes. Heavy-tailed variables like `number_customer_service_calls` are also not Gaussian. Fast and stable, but not competitive here. |
| Support Vector Machine | Second-best (F1 0.743, MCC 0.713) and the **highest AUC (0.905)** — the RBF kernel models the non-linear boundary well and ranks customers by risk better than any other model. At the default threshold it is slightly conservative (recall 66.7 %, 47 churners missed); lowering the threshold in the app trades precision for recall. Slower to train/predict than the tree and less interpretable. |
| **Overall, Winner for your dataset?** | **Decision Tree** — highest Accuracy, Precision, F1 and MCC (MCC is the most reliable single metric for this imbalanced data). It is also fully interpretable, so its rules can be handed to the retention team. If the goal is *ranking* customers by churn risk for a fixed-size campaign, **SVM** (best AUC) is the better choice. |

---

## Streamlit app features

The deployed app (`app.py`) provides:
1. **Dataset upload (CSV)** — upload test data containing the 19 feature columns and `churn`; a bundled `test_data.csv` is available for a one-click demo and download. Uploaded files are validated (missing columns, non-numeric values, NaNs, non-binary target).
2. **Model selection dropdown** — choose any of the five saved models.
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC displayed as metric cards.
4. **Confusion matrix and classification report** — heatmap plus per-class precision/recall/F1 table, and the ROC curve.
5. **Extras** — side-by-side comparison table and bar chart of all models, overlaid ROC curves, adjustable decision threshold, row-level predictions with misclassification filter, and CSV downloads.

The app **only loads the saved models** from `model/` and scores the uploaded rows; it never retrains. At the default threshold of 0.50 it reproduces the table above exactly on `test_data.csv`.

## How to run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_models.py      # optional - re-creates model/*.joblib, metrics.json, test_data.csv
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Go to https://streamlit.io/cloud → sign in with GitHub → **New app**.
3. Select the repository, branch `main`, main file `app.py`.
4. Under *Advanced settings*, choose **Python 3.12**, then **Deploy**.

## Tech stack

Python 3.12 · scikit-learn 1.9 · pandas · NumPy · Matplotlib · Seaborn · Streamlit · joblib
