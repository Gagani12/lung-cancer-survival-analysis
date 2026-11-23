# CONFIDENCE INTERVALS

print("\n========== 95% CONFIDENCE INTERVAL ==========")

survival = df["Survival_Months"].dropna()

n = len(survival)

mean_value = survival.mean()

std_value = survival.std(ddof=1)

standard_error = std_value / np.sqrt(n)

# 95% Confidence Interval
ci_lower, ci_upper = stats.t.interval(
    confidence=0.95,
    df=n-1,
    loc=mean_value,
    scale=standard_error
)

print("Sample Size:", n)

print("Mean Survival Months:", mean_value)

print("95% CI Lower:", ci_lower)

print("95% CI Upper:", ci_upper)

# GENDER-WISE CONFIDENCE INTERVALS

gender_ci_results = []

for gender, group in df.groupby("Gender"):

    values = group["Survival_Months"].dropna()

    n = len(values)

    mean_value = values.mean()

    se = stats.sem(values)

    lower, upper = stats.t.interval(
        confidence=0.95,
        df=n-1,
        loc=mean_value,
        scale=se
    )

    gender_ci_results.append({

        "Gender": gender,

        "Sample_Size": n,

        "Mean_Survival": mean_value,

        "Lower_CI": lower,

        "Upper_CI": upper

    })

gender_ci = pd.DataFrame(
    gender_ci_results
)

print("\nGender-wise Confidence Intervals:")

print(gender_ci)

gender_ci.to_csv(
    OUTPUT_DIR / "gender_confidence_intervals.csv",
    index=False
)

# CANCER STAGE CONFIDENCE INTERVALS

stage_ci_results = []

for stage, group in df.groupby("Cancer_Stage"):

    values = group["Survival_Months"].dropna()

    n = len(values)

    mean_value = values.mean()

    se = stats.sem(values)

    lower, upper = stats.t.interval(
        confidence=0.95,
        df=n-1,
        loc=mean_value,
        scale=se
    )

    stage_ci_results.append({

        "Cancer_Stage": stage,

        "Sample_Size": n,

        "Mean_Survival": mean_value,

        "Lower_CI": lower,

        "Upper_CI": upper

    })

stage_ci = pd.DataFrame(
    stage_ci_results
)

print("\nStage-wise Confidence Intervals:")

print(stage_ci)

stage_ci.to_csv(
    OUTPUT_DIR / "stage_confidence_intervals.csv",
    index=False
)

# CONFIDENCE INTERVAL PLOT

plt.figure(figsize=(9, 6))

plt.errorbar(
    x=stage_ci["Cancer_Stage"],
    y=stage_ci["Mean_Survival"],
    yerr=[
        stage_ci["Mean_Survival"] - stage_ci["Lower_CI"],
        stage_ci["Upper_CI"] - stage_ci["Mean_Survival"]
    ],
    fmt="o",
    capsize=5
)

plt.title(
    "95% Confidence Intervals of Survival Months by Cancer Stage"
)

plt.xlabel("Cancer Stage")

plt.ylabel("Mean Survival Months")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confidence_interval_plot.png",
    dpi=300
)

plt.close()

# MODEL VALIDATION

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

print("\n========== CROSS VALIDATION ==========")

# Prepare features (X) and target (y) based on your earlier dataset
# Using the predictors highlighted in your regression section
X = df[["Age", "BMI", "Tumor_Size_cm", "Cigarettes_Per_Day", "Years_Smoking"]].copy()
y = df["Survival_Months"].copy()

# Convert categorical predictors using pandas dummies to match requirements
X = pd.get_dummies(X, drop_first=True)

# Initialize models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit models
linear_model.fit(X, y)
rf_model.fit(X, y)

# 5-Fold Cross Validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Linear Regression CV
linear_cv_scores = cross_val_score(
    linear_model,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print("\nLinear Regression CV Scores:")
print(linear_cv_scores)
print(
    "Mean CV R2:",
    linear_cv_scores.mean()
)
print(
    "CV Standard Deviation:",
    linear_cv_scores.std()
)

# Random Forest CV
rf_cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print("\nRandom Forest CV Scores:")
print(rf_cv_scores)
print(
    "Mean CV R2:",
    rf_cv_scores.mean()
)
print(
    "CV Standard Deviation:",
    rf_cv_scores.std()
)

# RESIDUAL ANALYSIS

# Using the fitted values and residuals from the regression model
y_pred = regression_model.fittedvalues
residuals = regression_model.resid

print("\n========== RESIDUAL ANALYSIS ==========")

print("Mean Residual:", residuals.mean())

print("Residual Standard Deviation:", residuals.std())

# Residual plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Survival Months")

plt.ylabel("Residuals")

plt.title("Residual Plot - Linear Regression")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "regression_residual_plot.png",
    dpi=300
)

plt.close()

# FINAL MODEL VALIDATION SUMMARY
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate predictions to calculate metrics
y_pred_linear = linear_model.predict(X)
y_pred_rf = rf_model.predict(X)

# Calculate metrics for Linear Regression
mae = mean_absolute_error(y, y_pred_linear)
rmse = np.sqrt(mean_squared_error(y, y_pred_linear))
r2 = r2_score(y, y_pred_linear)

# Calculate metrics for Random Forest
rf_mae = mean_absolute_error(y, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y, y_pred_rf))
rf_r2 = r2_score(y, y_pred_rf)

validation_results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "Test_MAE": [
        mae,
        rf_mae
    ],

    "Test_RMSE": [
        rmse,
        rf_rmse
    ],

    "Test_R2": [
        r2,
        rf_r2
    ],

    "Mean_CV_R2": [
        linear_cv_scores.mean(),
        rf_cv_scores.mean()
    ],

    "CV_Std": [
        linear_cv_scores.std(),
        rf_cv_scores.std()
    ]
})

print("\n========== MODEL VALIDATION SUMMARY ==========")

print(validation_results)

validation_results.to_csv(
    OUTPUT_DIR / "model_validation_summary.csv",
    index=False
)

print("\nAll model validation results saved!")

