# SAVE STATISTICAL RESULTS
results = [
    {
        "Test": "Welch Independent t-test",
        "Comparison": "Survival Months by Gender",
        "Statistic": t_stat,
        "P_Value": stats.ttest_ind(
            male,
            female,
            equal_var=False
        ).pvalue
    },
    {
        "Test": "One-way ANOVA",
        "Comparison": "Survival Months by Cancer Stage",
        "Statistic": f_stat,
        "P_Value": stats.f_oneway(
            *stage_groups.values()
        ).pvalue
    },
    {
        "Test": "Mann-Whitney U",
        "Comparison": "Survival Months by Gender",
        "Statistic": u_stat,
        "P_Value": stats.mannwhitneyu(
            male,
            female,
            alternative="two-sided"
        ).pvalue
    },
    {
        "Test": "Kruskal-Wallis",
        "Comparison": "Survival Months by Cancer Stage",
        "Statistic": h_stat,
        "P_Value": stats.kruskal(
            *stage_groups.values()
        ).pvalue
    }
]

results_df = pd.DataFrame(results)

results_df["Decision_at_0.05"] = np.where(
    results_df["P_Value"] < 0.05,
    "Reject H0",
    "Fail to reject H0"
)

print("\n========== FINAL RESULTS SUMMARY ==========")

print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "hypothesis_testing_results.csv",
    index=False
)

print("\nStatistical results saved successfully!")
