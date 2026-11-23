# NORMALITY TESTING
print("\n========== NORMALITY TESTING ==========")

normality_columns = [
    "Age",
    "BMI",
    "Tumor_Size_cm",
    "Survival_Months"
]

for column in normality_columns:

    values = df[column].dropna()

    # Shapiro-Wilk test is computationally practical
    # on a sample for large datasets.
    if len(values) > 5000:
        values = values.sample(
            n=5000,
            random_state=42
        )

    statistic, p_value = stats.shapiro(values)

    print(f"\nVariable: {column}")
    print("Shapiro-Wilk Statistic:", statistic)
    print("P-value:", p_value)

    if p_value > 0.05:
        print("Result: No significant evidence against normality.")
    else:
        print("Result: Evidence against normality.")


# Q-Q PLOTS
for column in [
    "Age",
    "BMI",
    "Tumor_Size_cm",
    "Survival_Months"
]:

    plt.figure(figsize=(6, 6))

    stats.probplot(
        df[column].dropna(),
        dist="norm",
        plot=plt
    )

    plt.title(f"Q-Q Plot of {column}")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{column}_qqplot.png",
        dpi=300
    )

    plt.close()

print("Q-Q plots saved successfully!")


# LEVENE'S TEST

print("\n========== LEVENE'S TEST ==========")

male_survival = df.loc[
    df["Gender"] == "Male",
    "Survival_Months"
].dropna()

female_survival = df.loc[
    df["Gender"] == "Female",
    "Survival_Months"
].dropna()

levene_stat, levene_p = stats.levene(
    male_survival,
    female_survival,
    center="median"
)

print("Levene Statistic:", levene_stat)
print("P-value:", levene_p)

if levene_p > 0.05:
    print("No significant evidence of unequal variances.")
else:
    print("Evidence of unequal variances.")


