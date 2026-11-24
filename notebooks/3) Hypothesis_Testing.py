# INDEPENDENT SAMPLES T-TEST
print("\n========== INDEPENDENT T-TEST ==========")

male = df.loc[
    df["Gender"] == "Male",
    "Survival_Months"
].dropna()

female = df.loc[
    df["Gender"] == "Female",
    "Survival_Months"
].dropna()

print("\nGroup Sizes:")
print("Male:", len(male))
print("Female:", len(female))

print("\nMean Survival Months:")
print("Male:", male.mean())
print("Female:", female.mean())

# Welch's independent samples t-test
t_stat, p_value = stats.ttest_ind(
    male,
    female,
    equal_var=False
)

print("\nWelch's t-test Statistic:", t_stat)
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print(
        "Decision: Reject H0. "
        "There is evidence of a difference in mean survival months."
    )
else:
    print(
        "Decision: Fail to reject H0. "
        "There is insufficient evidence of a difference in mean survival months."
    )


# ONE-WAY ANOVA

print("\n========== ONE-WAY ANOVA ==========")

stage_groups = {}

for stage in sorted(df["Cancer_Stage"].dropna().unique()):

    stage_groups[stage] = df.loc[
        df["Cancer_Stage"] == stage,
        "Survival_Months"
    ].dropna()

    print(
        stage,
        "Count:",
        len(stage_groups[stage]),
        "Mean:",
        stage_groups[stage].mean()
    )

# Run one-way ANOVA
f_stat, p_value = stats.f_oneway(
    *stage_groups.values()
)

print("\nANOVA F-statistic:", f_stat)
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print(
        "Decision: Reject H0. "
        "There is evidence that at least one stage mean differs."
    )
else:
    print(
        "Decision: Fail to reject H0. "
        "There is insufficient evidence that the stage means differ."
    )


# ANOVA VARIANCE ASSUMPTION

print("\n========== ANOVA: LEVENE'S TEST ==========")

levene_stat, levene_p = stats.levene(
    *stage_groups.values(),
    center="median"
)

print("Levene Statistic:", levene_stat)
print("P-value:", levene_p)

if levene_p > 0.05:
    print(
        "No significant evidence of unequal variances."
    )
else:
    print(
        "Evidence of unequal variances. "
        "Consider Welch's ANOVA."
    )


#  ANOVA RESIDUAL NORMALITY

anova_model = ols(
    "Survival_Months ~ C(Cancer_Stage)",
    data=df
).fit()

residuals = anova_model.resid

# Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(
    residuals
)

print("\n========== ANOVA RESIDUAL NORMALITY ==========")

print("Shapiro Statistic:", shapiro_stat)
print("P-value:", shapiro_p)

if shapiro_p > 0.05:
    print(
        "No significant evidence against residual normality."
    )
else:
    print(
        "Evidence against residual normality."
    )

# Residual Q-Q plot
plt.figure(figsize=(6, 6))

stats.probplot(
    residuals,
    dist="norm",
    plot=plt
)

plt.title("Q-Q Plot of ANOVA Residuals")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "anova_residuals_qqplot.png",
    dpi=300
)

plt.close()


# MANN-WHITNEY U TEST

print("\n========== MANN-WHITNEY U TEST ==========")

u_stat, p_value = stats.mannwhitneyu(
    male,
    female,
    alternative="two-sided"
)

print("Mann-Whitney U Statistic:", u_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print(
        "Decision: Reject H0. "
        "There is evidence of a difference in the distributions."
    )
else:
    print(
        "Decision: Fail to reject H0. "
        "There is insufficient evidence of a distributional difference."
    )


# KRUSKAL-WALLIS TEST

print("\n========== KRUSKAL-WALLIS TEST ==========")
h_stat, p_value = stats.kruskal(
    *stage_groups.values()
)

print("Kruskal-Wallis H Statistic:", h_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print(
        "Decision: Reject H0. "
        "There is evidence of a difference among the group distributions."
    )
else:
    print(
        "Decision: Fail to reject H0. "
        "There is insufficient evidence of a difference among the group distributions."
    )
