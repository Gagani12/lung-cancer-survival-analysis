# Lung Cancer Patient Data – Advanced Statistical Analysis

Advanced Statistics final group project. We analyse a lung cancer patient dataset in Python to find out which patient and clinical factors are linked to **survival time**, check our statistical assumptions properly, and model the **monthly trend in diagnoses**.

All analysis is in one Jupyter notebook: `Stat_Group_Project_completed.ipynb` (made in Google Colab).

---

## Research Questions

1. Does mean survival time (`Survival_Months`) differ between **male and female** patients?
2. Does survival time differ across **cancer stages (I–IV)**?
3. Can patient characteristics and clinical factors explain the variation in survival months? (multiple linear regression)
4. Are there natural **patient groups** in the data? (PCA + K-Means clustering)
5. How are monthly diagnosis counts changing over time, and which **ARIMA / SARIMA** model forecasts them best?

---

## Dataset

| Item | Detail |
|---|---|
| File | `lung_cancer_dataset.csv` |
| Records | 2,000 patients (`LC-0001` … `LC-2000`) |
| Variables | 41 |
| Diagnosis period | 2015 – 2026 (monthly series: Jan 2015 – Jul 2026) |
| Coverage | 60 countries across 6 WHO regions |
| Missing values | 0 |
| Duplicate rows / IDs | 0 / 0 |

**Variable groups**

- **Demographics:** Age, Gender, Country, WHO_Region, BMI
- **Smoking & exposure:** Smoking_Status, Cigarettes_Per_Day, Years_Smoking, Secondhand_Smoke, Occupational_Hazard, Air_Pollution_Exposure, Asbestos_Exposure, Radon_Exposure, Alcohol_Use
- **History & lifestyle:** Family_History, Chronic_Lung_Disease, Previous_Cancer_History, Genetic_Mutation, Exercise_Frequency
- **Symptoms (Yes/No):** Coughing, Shortness_of_Breath, Chest_Pain, Coughing_Blood, Fatigue, Weight_Loss, Wheezing, Recurrent_Infections, Swallowing_Difficulty, Finger_Clubbing
- **Clinical:** Cancer_Type, NSCLC_Subtype, Cancer_Stage, Tumor_Size_cm, Metastasis, Diagnosis_Method, Treatment
- **Outcomes:** `Survival_Months` (1–110, mean 31.76), `Survived` (Yes 756 / No 1,244)

---

## Methods

| # | Section | What we did |
|---|---|---|
| 1 | **Data cleaning** | Checked duplicates and missing values, converted `Diagnosis_Date` to datetime, trimmed text categories, checked for invalid values (negative ages, BMI ≤ 0, etc.), saved a cleaned dataset |
| 2 | **Exploratory data analysis** | Descriptive statistics, histograms of numeric variables, bar charts of categorical variables, correlation heatmap |
| 3 | **Assumption checks** | Shapiro–Wilk normality tests, Q-Q plots, Levene's test for equal variances, normality of ANOVA residuals |
| 4 | **Hypothesis testing** | Welch's independent t-test (gender), one-way ANOVA (cancer stage), plus Mann–Whitney U and Kruskal–Wallis as non-parametric checks |
| 5 | **Regression** | Multiple linear regression (OLS) on `Survival_Months`, with residual-vs-fitted and residual Q-Q diagnostic plots |
| 6 | **PCA** | 25 standardised features (numeric + one-hot categoricals), scree plot, 2D PC plot |
| 7 | **Clustering** | K-Means for K = 2–7, chosen using the elbow method and silhouette score, then cluster profiles |
| 8 | **Time series** | Monthly diagnosis counts, rolling statistics, ADF stationarity test, differencing, ACF/PACF, comparison of 9 ARIMA and 3 SARIMA candidates on a time-ordered train/test split, residual checks (Ljung–Box), 95% forecast intervals |
| 9 | **Confidence intervals** | 95% t-intervals for mean survival: overall, by gender and by stage |
| 10 | **Model validation** | 5-fold cross-validation (R²) for Linear Regression and Random Forest Regressor, residual analysis |
| 11 | **Machine learning (extension)** | Survival classification pipeline with feature engineering (`Pack_Years`, `Total_Symptoms`), stratified split, Logistic Regression, Random Forest, Gradient Boosting and XGBoost, tuning with RandomizedSearchCV, ROC curves, confusion matrix, feature importance |

---

## Key Results

### Assumption checks
- Shapiro–Wilk rejected normality for Age, BMI, Tumor_Size_cm and Survival_Months (all p < 0.01).
- Levene's test showed unequal variances by gender (p = 0.034) and by stage (p < 0.001).
- Because of this we used **Welch's t-test** and confirmed every result with **non-parametric tests**.

### Hypothesis tests (α = 0.05)

| Test | Comparison | Statistic | p-value | Decision |
|---|---|---|---|---|
| Welch's t-test | Survival months by gender | t = −3.42 | 0.0006 | Reject H₀ |
| Mann–Whitney U | Survival months by gender | U = 429,113 | 0.0013 | Reject H₀ |
| One-way ANOVA | Survival months by stage | F = 1793.60 | < 0.001 | Reject H₀ |
| Kruskal–Wallis | Survival months by stage | H = 1543.02 | < 0.001 | Reject H₀ |

### 95% confidence intervals for mean survival (months)

| Group | n | Mean | 95% CI |
|---|---|---|---|
| All patients | 2000 | 31.76 | 30.71 – 32.80 |
| Female | 752 | 34.14 | 32.36 – 35.92 |
| Male | 1248 | 30.32 | 29.04 – 31.60 |
| Stage I | 577 | 59.72 | 58.27 – 61.18 |
| Stage II | 487 | 35.51 | 34.31 – 36.72 |
| Stage III | 460 | 17.86 | 17.18 – 18.53 |
| Stage IV | 476 | 7.44 | 7.03 – 7.84 |

Survival falls sharply as stage increases, and the stage intervals do not overlap.

### Multiple linear regression
- Model: `Survival_Months ~ Age + BMI + Tumor_Size_cm + Cigarettes_Per_Day + Years_Smoking + Gender + Smoking_Status + Cancer_Stage + Metastasis + Treatment`
- **R² = 0.731, adjusted R² = 0.728**, F(19, 1980) = 283.2, p < 0.001
- **Cancer stage is the only significant predictor.** Compared with Stage I, survival is lower by about 24.8 months (Stage II), 42.1 months (Stage III) and 51.5 months (Stage IV), all p < 0.001.
- Once stage is in the model, gender, age, BMI, tumour size, smoking, metastasis and treatment are **not** significant (p > 0.05). So the gender difference in the t-test is most likely explained by differences in stage.

### PCA and clustering
- PC1 explains 19.69% and PC2 12.52% of variance (32.2% together), so the variance is spread across many dimensions.
- The best silhouette score was at **K = 2** (0.190):
  - **Cluster 0 (549 patients):** larger tumours (mean 6.9 cm), mostly Stage IV, median survival 8 months
  - **Cluster 1 (1,451 patients):** smaller tumours (mean 3.6 cm), mostly Stage I, median survival 38 months

### Time series (monthly diagnosis counts)
- 139 monthly observations. The ADF test showed the original series is non-stationary (p = 0.87). After first differencing it is stationary (p < 0.001), so **d = 1**.
- Train: Jan 2015 – Mar 2024 (111 months). Test: Apr 2024 – Jul 2026 (28 months).
- **Selected model: SARIMA(1,1,1)(0,1,1)₁₂**, which had the lowest test RMSE (6.08) and MAE (5.09) and the lowest AIC (475.65) of all 12 candidates. The best non-seasonal model was ARIMA(0,1,1), with RMSE 7.32.
- The model was refit on the full series to forecast future months with 95% intervals. The Ljung–Box p-value (0.0035) suggests some residual autocorrelation is left, so the forecast should be read as a planning estimate.

### Model validation (Survival_Months, numeric predictors only)

| Model | Mean 5-fold CV R² | CV SD |
|---|---|---|
| Linear Regression | 0.474 | 0.026 |
| Random Forest Regressor | 0.484 | 0.043 |

> Note: the "Test" MAE/RMSE/R² values in `model_validation_summary.csv` are calculated on the same data the models were trained on (in-sample). The cross-validated R² above is the fair measure of performance on new data.

---

## Repository Structure

```
├── Stat_Group_Project_completed.ipynb   # Full analysis notebook
├── lung_cancer_dataset.csv              # Raw dataset
├── outputs/                             # Generated when the notebook runs
│   ├── lung_cancer_cleaned.csv
│   ├── clustered_lung_cancer_data.csv
│   ├── regression_summary.txt
│   ├── regression_coefficients.csv
│   ├── cluster_evaluation.csv
│   ├── model_validation_summary.csv
│   ├── *_histogram.png, *_countplot.png, correlation_heatmap.png
│   ├── *_qqplot.png, anova_residuals_qqplot.png, regression_residual_plot.png
│   ├── pca_scree_plot.png, pca_2d_visualization.png
│   ├── elbow_method.png, silhouette_analysis.png, kmeans_cluster_visualization.png
│   ├── monthly_time_series.png, rolling_statistics.png, stationary_series.png
│   ├── acf_pacf_stationary_series.png, time_series_residual_diagnostics.png
│   ├── time_series_residual_acf.png, time_series_forecast.png
│   ├── confidence_interval_plot.png
│   └── ml_roc_curves.png, optimized_confusion_matrix.png, top_15_feature_importance.png
└── README.md
```

---

## How to Run

# Run locally
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn xgboost jupyter
jupyter notebook Stat_Group_Project_completed.ipynb
```
Change the dataset path in the first code cell to:
```python
df = pd.read_csv("lung_cancer_dataset.csv")
```

---

## Tech Stack

- **Python 3** (Jupyter / Google Colab)
- **pandas, NumPy:** data handling
- **Matplotlib, Seaborn:** visualisation
- **SciPy:** Shapiro–Wilk, Levene, t-test, ANOVA, Mann–Whitney, Kruskal–Wallis, confidence intervals
- **statsmodels:** OLS regression, ADF test, ACF/PACF, ARIMA/SARIMAX, Ljung–Box
- **scikit-learn:** PCA, K-Means, silhouette score, cross-validation, ML pipelines
- **XGBoost:** gradient-boosted classifier

---

## Limitations

- Several variables are not normally distributed and group variances differ, which is why we added non-parametric tests alongside the parametric ones.
- Linear regression residuals are not perfectly normal, so p-values should be read with some caution.
- The first two principal components explain only about 32% of the variance, and silhouette scores are low (about 0.19), so the clusters overlap.
- The SARIMA residuals still show some autocorrelation (Ljung–Box p < 0.05).
- The results show association, not causation.

---

## Team

| Name | Student ID | Contribution |
|---|---|---|
| Gagani Rathnayaka | 36508 | |
| | | |
| | | |
| | | |

---

*This project was completed for academic purposes as part of the Advanced Statistics module.*
