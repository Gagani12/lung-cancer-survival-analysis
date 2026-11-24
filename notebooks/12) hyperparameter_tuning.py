# ==========================================
# 3. MODEL OPTIMIZATION (RANDOMIZED SEARCH)
# ==========================================
# Optimize Random Forest & Gradient Boosting via RandomizedSearchCV
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_param_dist = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 6, 10, 15, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": ["balanced", "balanced_subsample", None]
}

rf_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=rf_param_dist,
    n_iter=20,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train, y_train)
best_model = rf_search.best_estimator_

print("\n========== OPTIMIZED MODEL RESULTS ==========")
print("Best Hyperparameters:", rf_search.best_params_)
print(f"Best CV ROC-AUC: {rf_search.best_score_:.4f}")

# Evaluate Optimized Model on Test Set
y_pred_opt = best_model.predict(X_test)
y_prob_opt = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_opt, target_names=["Not Survived", "Survived"]))

# Confusion Matrix Visualization
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_opt, display_labels=["No", "Yes"], cmap="Blues", ax=ax
)
plt.title("Confusion Matrix - Optimized Random Forest")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "optimized_confusion_matrix.png", dpi=300)
plt.show()
