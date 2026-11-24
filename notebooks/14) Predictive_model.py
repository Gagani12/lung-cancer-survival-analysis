
# 5. SECONDARY PREDICTIVE MODEL: SURVIVAL MONTHS (REGRESSION)

y_reg = df["Survival_Months"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
])

reg_pipeline.fit(X_train_r, y_train_r)
y_pred_r = reg_pipeline.predict(X_test_r)

mae = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print("\n========== SURVIVAL MONTHS REGRESSION PREDICTIVE ANALYTICS ==========")
print(f"MAE  : {mae:.2f} months")
print(f"RMSE : {rmse:.2f} months")
print(f"R²   : {r2:.4f}")
