# PREPARE A MONTHLY TIME SERIES

import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Parse dates and remove records without a valid diagnosis date
time_df = df.copy()
time_df["Diagnosis_Date"] = pd.to_datetime(
    time_df["Diagnosis_Date"], errors="coerce"
)
time_df = time_df.dropna(subset=["Diagnosis_Date"]).copy()
time_df = time_df.sort_values("Diagnosis_Date")

if time_df.empty:
    raise ValueError("No valid Diagnosis_Date values are available.")

# Count diagnosis records for each calendar month.
# MS = month start, missing months are filled with 0 to keep equal intervals.
monthly_series = (
    time_df.set_index("Diagnosis_Date")
    .resample("MS")
    .size()
    .asfreq("MS", fill_value=0)
)
monthly_series.name = "Patient_Count"

print("========== TIME SERIES INFORMATION ==========")
print("First month:", monthly_series.index.min())
print("Last month:", monthly_series.index.max())
print("Number of monthly observations:", len(monthly_series))
print("Missing observations:", monthly_series.isna().sum())
print("Frequency:", monthly_series.index.freq)

# Confirm equal monthly intervals
month_gaps = monthly_series.index.to_series().diff().dropna()
print("All observations are equally spaced:",
      bool((month_gaps == pd.offsets.MonthBegin(1)).all()))

print("\\nMonthly diagnosis counts:")
print(monthly_series.head())
print(monthly_series.tail())

monthly_series.to_csv(OUTPUT_DIR / "monthly_diagnosis_counts.csv")

# VISUALIZE THE MONTHLY SERIES

plt.figure(figsize=(13, 5))
plt.plot(monthly_series.index, monthly_series.values, marker="o", markersize=3)
plt.title("Monthly Lung Cancer Diagnosis Records")
plt.xlabel("Month")
plt.ylabel("Number of Diagnosis Records")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_time_series.png", dpi=300)
plt.show()

# Rolling mean and standard deviation help identify changes over time
rolling_window = min(12, max(3, len(monthly_series) // 4))
rolling_mean = monthly_series.rolling(rolling_window).mean()
rolling_std = monthly_series.rolling(rolling_window).std()

plt.figure(figsize=(13, 5))
plt.plot(monthly_series, label="Monthly count")
plt.plot(rolling_mean, label=f"{rolling_window}-month rolling mean")
plt.plot(rolling_std, label=f"{rolling_window}-month rolling SD")
plt.title("Monthly Series with Rolling Mean and Standard Deviation")
plt.xlabel("Month")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rolling_statistics.png", dpi=300)
plt.show()

print("Visual inspection: examine the plots for a long-term trend, repeated yearly "
      "seasonality, and irregular fluctuations.")

# AUGMENTED DICKEY-FULLER (ADF) TEST
def run_adf(series, label):
    series = series.dropna()
    print(f"\\nADF test – {label}")
    result = adfuller(series, autolag="AIC")
    print("ADF statistic:", result[0])
    print("p-value:", result[1])
    print("Used lags:", result[2])
    print("Number of observations:", result[3])
    print("Critical values:", result[4])
    if result[1] < 0.05:
        print("Decision: Reject the unit-root null hypothesis. Series is stationary.")
        return True
    else:
        print("Decision: Fail to reject the unit-root null hypothesis. "
              "Series may be non-stationary.")
        return False

is_stationary = run_adf(monthly_series, "Original monthly series")

# APPLY DIFFERENCING ONLY WHEN REQUIRED
stationary_series = monthly_series.copy()
d = 0
stationary = is_stationary

while not stationary and d < 2:
    d += 1
    stationary_series = monthly_series.diff(d).dropna() if d == 2 else monthly_series.diff().dropna()
    # For d=2, difference the already first-differenced series once more
    if d == 2:
        stationary_series = monthly_series.diff().diff().dropna()
    stationary = run_adf(stationary_series, f"After differencing d={d}")

if not stationary:
    print("\\nWarning: The series did not become stationary after second differencing.")
else:
    print(f"\\nSelected non-seasonal differencing order d = {d}")

plt.figure(figsize=(12, 4))
plt.plot(stationary_series)
plt.title(f"Series after differencing (d={d})")
plt.xlabel("Month")
plt.ylabel("Differenced patient count")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "stationary_series.png", dpi=300)
plt.show()

# ACF AND PACF OF THE STATIONARY SERIES
if len(stationary_series) >= 8:
    max_lags = min(24, max(1, len(stationary_series) // 2 - 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(stationary_series, lags=max_lags, ax=axes[0])
    plot_pacf(stationary_series, lags=max_lags, ax=axes[1], method="ywm")
    axes[0].set_title("ACF – Stationary Series")
    axes[1].set_title("PACF – Stationary Series")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "acf_pacf_stationary_series.png", dpi=300)
    plt.show()
else:
    print("Not enough observations to produce reliable ACF/PACF plots.")

print("Use the ACF and PACF patterns as guidance for candidate p and q values. "
      "The next step compares nearby model orders using training fit and test forecasts.")

# CREATE TIME-ORDERED TRAINING AND TEST SETS
if len(monthly_series) < 18:
    raise ValueError("At least 18 monthly observations are recommended for model comparison.")

test_size = max(3, int(round(len(monthly_series) * 0.20)))
test_size = min(test_size, len(monthly_series) - 12)

train_series = monthly_series.iloc[:-test_size]
test_series = monthly_series.iloc[-test_size:]

print("Training observations:", len(train_series))
print("Testing observations:", len(test_series))
print("Training period:", train_series.index.min(), "to", train_series.index.max())
print("Testing period:", test_series.index.min(), "to", test_series.index.max())

# Find d on the training set only, avoiding information leakage from the test period
train_d = 0
train_stationary = run_adf(train_series, "Training series")
train_diff = train_series.copy()
while not train_stationary and train_d < 2:
    train_diff = train_diff.diff().dropna()
    train_d += 1
    train_stationary = run_adf(train_diff, f"Training series after d={train_d}")

print("Training-selected d:", train_d)

# Compare a small set of ARIMA candidates
candidate_results = []
candidate_fits = {}

for p in range(3):
    for q in range(3):
        order = (p, train_d, q)
        try:
            fitted = ARIMA(
                train_series,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit()
            predictions = fitted.forecast(steps=len(test_series))
            rmse = np.sqrt(mean_squared_error(test_series, predictions))
            mae = mean_absolute_error(test_series, predictions)

            candidate_results.append({
                "Model": "ARIMA",
                "Order": str(order),
                "Seasonal_Order": "-",
                "AIC": fitted.aic,
                "BIC": fitted.bic,
                "Test_RMSE": rmse,
                "Test_MAE": mae
            })
            candidate_fits[("ARIMA", order, None)] = fitted
        except Exception as error:
            print(f"ARIMA{order} could not be fitted: {error}")

# Check whether there is enough data to assess annual (12-month) seasonality.
# At least 3 years of monthly observations are required for this comparison.
seasonal_possible = len(train_series) >= 36
if seasonal_possible:
    seasonal_orders = [
        (1, 0, 1, 12),
        (0, 1, 1, 12),
        (1, 1, 0, 12)
    ]
    for p, D, q, s in seasonal_orders:
        seasonal_order = (p, D, q, s)
        try:
            fitted = SARIMAX(
                train_series,
                order=(1, train_d, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            predictions = fitted.forecast(steps=len(test_series))
            rmse = np.sqrt(mean_squared_error(test_series, predictions))
            mae = mean_absolute_error(test_series, predictions)

            candidate_results.append({
                "Model": "SARIMA",
                "Order": str((1, train_d, 1)),
                "Seasonal_Order": str(seasonal_order),
                "AIC": fitted.aic,
                "BIC": fitted.bic,
                "Test_RMSE": rmse,
                "Test_MAE": mae
            })
            candidate_fits[("SARIMA", (1, train_d, 1), seasonal_order)] = fitted
        except Exception as error:
            print(f"SARIMA seasonal order {seasonal_order} could not be fitted: {error}")
else:
    print("SARIMA comparison skipped: fewer than 36 training months. "
          "There is not enough history to assess repeated annual seasonality reliably.")

if not candidate_results:
    raise RuntimeError("No candidate time-series model could be fitted.")

candidate_table = pd.DataFrame(candidate_results).sort_values(
    ["Test_RMSE", "AIC"]
).reset_index(drop=True)

print("\\nCandidate model comparison:")
print(candidate_table.to_string(index=False))
candidate_table.to_csv(OUTPUT_DIR / "time_series_model_comparison.csv", index=False)

# SELECT THE BEST VALIDATION MODEL AND CHECK RESIDUALS
best_row = candidate_table.iloc[0]
best_model_type = best_row["Model"]

if best_model_type == "ARIMA":
    best_order = eval(best_row["Order"])
    best_seasonal_order = None
    selected_key = ("ARIMA", best_order, None)
    selected_fit = candidate_fits[selected_key]
else:
    best_order = eval(best_row["Order"])
    best_seasonal_order = eval(best_row["Seasonal_Order"])
    selected_key = ("SARIMA", best_order, best_seasonal_order)
    selected_fit = candidate_fits[selected_key]

residuals = pd.Series(selected_fit.resid).dropna()

print("Selected candidate:", best_model_type)
print("ARIMA order:", best_order)
print("Seasonal order:", best_seasonal_order)
print("Residual mean:", residuals.mean())

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(residuals)
axes[0].axhline(0, linestyle="--")
axes[0].set_title("Selected Model Residuals")
axes[1].hist(residuals, bins=20)
axes[1].set_title("Residual Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "time_series_residual_diagnostics.png", dpi=300)
plt.show()

resid_lags = min(12, max(1, len(residuals) // 5))
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(residuals, lags=min(resid_lags, max(1, len(residuals)//2 - 1)), ax=ax)
ax.set_title("ACF of Model Residuals")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "time_series_residual_acf.png", dpi=300)
plt.show()

if len(residuals) > 12:
    lb_result = acorr_ljungbox(residuals, lags=[min(12, len(residuals)//3)], return_df=True)
    print("\\nLjung-Box test:")
    print(lb_result)
    lb_pvalue = lb_result["lb_pvalue"].iloc[0]
    if lb_pvalue > 0.05:
        print("Residual decision: no significant remaining autocorrelation detected.")
    else:
        print("Residual decision: significant autocorrelation remains; model may need improvement.")
else:
    lb_pvalue = np.nan
    print("Not enough residual observations for a reliable Ljung-Box test.")

print("A good model should have residuals fluctuating randomly around zero "
      "and a Ljung-Box p-value greater than 0.05.")

# FINAL MODEL SELECTION
# Primary criterion: lowest holdout RMSE. AIC/BIC are used as supporting
# measures of fit-versus-complexity, not as substitutes for test performance.
candidate_table = candidate_table.sort_values(
    ["Test_RMSE", "AIC"]
).reset_index(drop=True)

print("Ranked candidate models:")
print(candidate_table.to_string(index=False))

final_row = candidate_table.iloc[0]
final_model_type = final_row["Model"]
final_order = eval(final_row["Order"])
final_seasonal_order = (
    eval(final_row["Seasonal_Order"])
    if final_model_type == "SARIMA" else None
)

print("\\nFinal selected model:", final_model_type)
print("Order:", final_order)
print("Seasonal order:", final_seasonal_order)
print("Test RMSE:", final_row["Test_RMSE"])
print("Test MAE:", final_row["Test_MAE"])
print("AIC:", final_row["AIC"])
print("BIC:", final_row["BIC"])

# REFIT THE SELECTED MODEL ON THE FULL SERIES AND FORECAST
if final_model_type == "ARIMA":
    final_model = ARIMA(
        monthly_series,
        order=final_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()
else:
    final_model = SARIMAX(
        monthly_series,
        order=final_order,
        seasonal_order=final_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

print(final_model.summary())

forecast_steps = 12  # Forecast the next 12 months
forecast_result = final_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

forecast_table = pd.DataFrame({
    "Forecast": forecast_mean,
    "Lower_95_CI": forecast_ci.iloc[:, 0],
    "Upper_95_CI": forecast_ci.iloc[:, 1]
})
print("\\nNext 12-month forecast with 95% confidence intervals:")
print(forecast_table)

forecast_table.to_csv(OUTPUT_DIR / "time_series_forecast_12_months.csv")

plt.figure(figsize=(13, 6))
plt.plot(monthly_series.index, monthly_series, label="Observed")
plt.plot(forecast_mean.index, forecast_mean, label="Forecast", marker="o")
plt.fill_between(
    forecast_mean.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.2,
    label="95% confidence interval"
)
plt.title(f"12-Month Forecast using {final_model_type}")
plt.xlabel("Month")
plt.ylabel("Number of Diagnosis Records")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "time_series_forecast.png", dpi=300)
plt.show()

# AUTOMATIC SUMMARY TO SUPPORT THE REAL-WORLD INTERPRETATION
print("========== TIME SERIES ANALYSIS SUMMARY ==========")
print("Time interval: Monthly")
print("Number of observations:", len(monthly_series))
print("Differencing order d (training data):", train_d)
print("Selected model:", final_model_type)
print("ARIMA order:", final_order)
print("Seasonal order:", final_seasonal_order)
print("Holdout RMSE:", round(float(final_row["Test_RMSE"]), 3))
print("Holdout MAE:", round(float(final_row["Test_MAE"]), 3))
print("AIC:", round(float(final_row["AIC"]), 3))
print("BIC:", round(float(final_row["BIC"]), 3))
print("Ljung-Box p-value:", lb_pvalue)

if final_model_type == "SARIMA":
    print("\\nConclusion: SARIMA was selected because it achieved the best holdout "
          "forecast RMSE among the tested candidates. Its seasonal component can "
          "represent a repeating 12-month pattern.")
else:
    print("\\nConclusion: ARIMA was selected because it achieved the best holdout "
          "forecast RMSE among the tested candidates. The tested evidence did not "
          "show a sufficient forecasting improvement from the seasonal SARIMA candidates.")

print("\\nInterpretation guide:")
print("- Forecast values estimate the monthly number of diagnosis records.")
print("- The 95% confidence interval represents forecast uncertainty.")
print("- Wider intervals indicate greater uncertainty further into the future.")
print("- Use the forecast as a planning estimate, not as a causal explanation.")
print("\\nImportant: the final ARIMA/SARIMA decision is data-dependent. "
      "Use the printed comparison table and diagnostics, not a pre-assumed model.")

