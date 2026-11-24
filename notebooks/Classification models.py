
# 2. BASELINE CLASSIFICATION MODELS

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="logloss",
        random_state=42
    )
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

plt.figure(figsize=(8, 6))
ax = plt.gca()

for name, model in models.items():
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    
    # 5-Fold Cross-Validation ROC-AUC
    cv_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    
    # Fit on training data and evaluate on hold-out test set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    results.append({
        "Model": name,
        "CV_ROC_AUC": round(cv_auc, 4),
        "Test_Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1_Score": round(f1_score(y_test, y_pred), 4),
        "Test_ROC_AUC": round(roc_auc_score(y_test, y_prob), 4)
    })
    
    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)

plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.title("ROC Curve Comparison - Lung Cancer Survival Prediction")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ml_roc_curves.png", dpi=300)
plt.show()

results_df = pd.DataFrame(results).sort_values(by="Test_ROC_AUC", ascending=False)
print("\n========== BASELINE MODEL PERFORMANCE ==========")
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "baseline_model_comparison.csv", index=False)
