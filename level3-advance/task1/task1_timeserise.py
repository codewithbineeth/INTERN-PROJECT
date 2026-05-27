import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

print("=" * 70)
print(" ADVANCED LEVEL 3 TASK 1: TIME SERIES ANALYSIS & FORECASTING")
print("=" * 70)

# ==========================================
#  DATA ENGINEERING: GENERATING CLEAN TIME FREQUENCY SERIES
# ==========================================
print(" Step 1: Generating 3-Year Daily Company Sales Dataset...")
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
n = len(dates)

# Construct structural components: Base value + Constant upward trend + Seasonality + Noise
base_value = 100
linear_trend = np.linspace(0, 150, n)
annual_seasonality = 30 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
weekly_seasonality = 15 * np.sin(2 * np.pi * dates.dayofweek / 7)
random_noise = np.random.normal(0, 15, n)

# Accumulate into clean aggregate metric
sales_stream = base_value + linear_trend + annual_seasonality + weekly_seasonality + random_noise
df_daily = pd.DataFrame({"Sales": sales_stream}, index=dates)

# Downsample to Monthly averages for clean macroeconomic forecasting
df_monthly = df_daily.resample("ME").mean()
df_monthly.index.freq = "ME"
print(f"   Processed {len(df_monthly)} consecutive calendar months.")


# ==========================================
#  OBJECTIVE 1: TIME SERIES DECOMPOSITION
# ==========================================
print("\n Step 2: Decomposing Series into Trend, Seasonality, & Residuals...")
decomposition = seasonal_decompose(df_monthly["Sales"], model="additive")

# Capture and adjust built-in plot canvas scale layout safely
fig = decomposition.plot()
fig.set_size_inches(11, 8)
plt.suptitle("Time Series Decomposition Profiles", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

decomp_image = "time_series_decomposition.png"
plt.savefig(decomp_image, dpi=300)
plt.close()
print(f"    Saved Decomposition View: '{decomp_image}'")


# ==========================================
#  OBJECTIVE 2: SMOOTHING TECHNIQUES (MA & EXPONENTIAL)
# ==========================================
print("\n Step 3: Implementing Moving Average & Simple Exponential Smoothing...")

# 1. 3-Month Moving Average (Rolling Mean)
df_monthly["Moving_Average_3M"] = df_monthly["Sales"].rolling(window=3).mean()

# 2. Simple Exponential Smoothing (SES) with an explicit learning weight alpha = 0.3
ses_model = SimpleExpSmoothing(df_monthly["Sales"]).fit(smoothing_level=0.3, optimized=False)
df_monthly["SES"] = ses_model.fittedvalues

# Plot the smoothing comparisons
plt.figure(figsize=(11, 5))
plt.plot(df_monthly.index, df_monthly["Sales"], label="Original Sales", color="black", alpha=0.4, lw=1.5)
plt.plot(df_monthly.index, df_monthly["Moving_Average_3M"], label="3-Month Moving Average", color="orange", lw=2)
plt.plot(df_monthly.index, df_monthly["SES"], label="Exponential Smoothing (α=0.3)", color="green", lw=2)

plt.title("Time Series Smoothing Comparisons", fontsize=12, fontweight="bold")
plt.xlabel("Timeline Date")
plt.ylabel("Sales Volume")
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.5)

smoothing_image = "time_series_smoothing.png"
plt.savefig(smoothing_image, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Saved Smoothing Comparison: '{smoothing_image}'")


# ==========================================
#  OBJECTIVES 3 & 4: ARIMA MODELING, FORECAST & RMSE EVALUATION
# ==========================================
print("\n Step 4: Structuring ARIMA Predictive Modeling...")

# Partition into Training context and final 6 months for Validation Testing
train_subset = df_monthly.iloc[:-6]
test_subset = df_monthly.iloc[-6:]

# Fitting an ARIMA(1, 1, 1) model
# p=1 (Autoregressive lag), d=1 (First-order differencing), q=1 (Moving Average window)
arima_model = ARIMA(train_subset["Sales"], order=(1, 1, 1))
arima_fit = arima_model.fit()

# Generate mathematical step forecast projections
forecast_steps = len(test_subset)
forecast_series = arima_fit.forecast(steps=forecast_steps)

# Calculate Evaluation Accuracy Metric: Root Mean Squared Error (RMSE)
rmse_score = np.sqrt(mean_squared_error(test_subset["Sales"], forecast_series))
print(f"   -> Out-of-Sample Evaluation Complete.")
print(f"   -> Model Validation Performance (RMSE): {rmse_score:.4f} units")


# ==========================================
#  OBJECTIVE 5: VISUALIZE FORECAST VS ACTUALS
# ==========================================
print("\n Step 5: Visualizing Forecast Horizon Projections...")
plt.figure(figsize=(11, 5))

# Plot historical train data, true unseen validation points, and our model's guesses
plt.plot(train_subset.index, train_subset["Sales"], label="Historical Train Data", color="royalblue", lw=2)
plt.plot(test_subset.index, test_subset["Sales"], label="Actual Validation Data", color="black", marker="o", lw=1.5)
plt.plot(test_subset.index, forecast_series, label=f"ARIMA Forecast Line (RMSE: {rmse_score:.2f})", color="crimson", marker="x", linestyle="--", lw=2)

plt.title("Company Sales Forecasting Horizon via ARIMA(1,1,1)", fontsize=12, fontweight="bold")
plt.xlabel("Timeline Target")
plt.ylabel("Sales Value Scale")
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.5)

forecast_image = "time_series_forecast.png"
plt.savefig(forecast_image, dpi=300, bbox_inches="tight")
plt.close()

print(f"    Saved Forecast Horizon View: '{forecast_image}'")
print("\n" + "=" * 70)
print(" ADVANCED TIME SERIES WORKFLOW COMPLETED SUCCESSFULLY!")
print("=" * 70)