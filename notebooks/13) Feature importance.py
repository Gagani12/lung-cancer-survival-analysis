# ==========================================
# 4. PREDICTIVE ANALYTICS & FEATURE IMPORTANCE
# ==========================================
# Extract feature names after OneHotEncoding
ohe_features = best_model.named_steps["preprocessor"].named_transformers_["cat"]\
    .named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + ohe_features

importances = best_model.named_steps["classifier"].feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot Top 15 Predictors of Lung Cancer Survival
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(15), x="Importance", y="Feature", palette="viridis")
plt.title("Top 15 Predictive Features for Patient Survival")
plt.xlabel("Feature Importance Score")
plt.ylabel("Clinical / Demographic Feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_15_feature_importance.png", dpi=300)
plt.show()

feat_imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
