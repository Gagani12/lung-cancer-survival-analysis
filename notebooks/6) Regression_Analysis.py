# MULTIPLE LINEAR REGRESSION
print("\n========== REGRESSION ANALYSIS ==========")

# Import required library
import statsmodels.formula.api as smf

# Define regression model
regression_model = smf.ols(
    formula="""
    Survival_Months ~ Age + BMI + Tumor_Size_cm
    + Cigarettes_Per_Day + Years_Smoking
    + C(Gender)
    + C(Smoking_Status)
    + C(Cancer_Stage)
    + C(Metastasis)
    + C(Treatment)
    """,
    data=df
).fit()

# Display model summary
print(regression_model.summary())

# Save regression summary
with open(
    OUTPUT_DIR / "regression_summary.txt",
    "w"
) as file:

    file.write(regression_model.summary().as_text())

# Save regression coefficients
regression_coefficients = pd.DataFrame({
    "Coefficient": regression_model.params,
    "P_Value": regression_model.pvalues,
    "Lower_CI": regression_model.conf_int()[0],
    "Upper_CI": regression_model.conf_int()[1]
})

print("\nRegression Coefficients:")
print(regression_coefficients)

regression_coefficients.to_csv(
    OUTPUT_DIR / "regression_coefficients.csv"
)

# REGRESSION DIAGNOSTICS


# Residual vs Fitted plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=regression_model.fittedvalues,
    y=regression_model.resid
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "regression_residual_plot.png",
    dpi=300
)

plt.close()


# Regression residual Q-Q plot
plt.figure(figsize=(6, 6))

stats.probplot(
    regression_model.resid,
    dist="norm",
    plot=plt
)

plt.title("Regression Residual Q-Q Plot")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "regression_qqplot.png",
    dpi=300
)

plt.close()

print("Regression diagnostic plots saved!")

