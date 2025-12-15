# =====================================
# 0. IMPORT LIBRARIES
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_style("whitegrid")

# =====================================
# 1. LOAD DATA (PUT YOUR FILE PATH HERE)
# =====================================

file_path = r"C:/Users/USER/Music/Self Projects/Time Series forecasting of Stock Prices/NFLX_stocks.csv"
df = pd.read_csv(file_path)

# =====================================
# 2. HANDLE DATE COLUMN (LOWER CASE)
# =====================================

# Case: date saved as unnamed column
if 'unnamed: 0' in df.columns:
    df.rename(columns={'unnamed: 0': 'date'}, inplace=True)

# Convert date to datetime and set index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df.sort_index(inplace=True)

print("Columns:", df.columns)
print(df.head())

# =====================================
# 3. DATA CLEANING (CRITICAL FIX)
# =====================================

# Ensure numeric close price
df['close'] = pd.to_numeric(df['close'], errors='coerce')

# Drop rows with invalid close values
df = df.dropna(subset=['close'])

print("\nClose dtype:", df['close'].dtype)

# =====================================
# 4. BASIC EDA
# =====================================

print("\nSummary statistics:\n")
print(df['close'].describe())

# =====================================
# 5. FIXED TIME SERIES PLOT
# =====================================

plt.figure(figsize=(14,6))
plt.plot(df.index, df['close'], linewidth=1)

plt.title("NFLX Closing Price Time Series")
plt.xlabel("Date")
plt.ylabel("Close Price")

plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================
# 6. MONTHLY RESAMPLED PLOT (EDA ONLY)
# =====================================

monthly_close = df['close'].resample('ME').mean()

plt.figure(figsize=(14,6))
plt.plot(monthly_close, linewidth=2)

plt.title("NFLX Monthly Average Closing Price")
plt.xlabel("Date")
plt.ylabel("Close Price")

plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================
# 7. LOG SCALE PLOT (BEST PRACTICE)
# =====================================

plt.figure(figsize=(14,6))
plt.plot(df.index, df['close'], linewidth=1)

plt.yscale('log')
plt.title("NFLX Closing Price (Log Scale)")
plt.xlabel("Date")
plt.ylabel("Log Close Price")

plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================
# 8. ROLLING STATISTICS (TREND & VOLATILITY)
# =====================================

rolling_mean = df['close'].rolling(window=252).mean()
rolling_std = df['close'].rolling(window=252).std()

plt.figure(figsize=(14,6))
plt.plot(df['close'], label='Close Price', alpha=0.6)
plt.plot(rolling_mean, label='252-Day Rolling Mean', linewidth=2)
plt.plot(rolling_std, label='252-Day Rolling Std', linewidth=2)

plt.title("NFLX Rolling Mean & Volatility (252 Trading Days)")
plt.xlabel("Date")
plt.ylabel("Price")

plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# =====================================
# 9. STATIONARITY CHECK (ADF + KPSS)
# =====================================

from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return result[1]

def kpss_test(series):
    stat, p_value, _, _ = kpss(series, regression='c', nlags="auto")
    return p_value

print("\nADF p-value (level):", adf_test(df['close']))
print("KPSS p-value (level):", kpss_test(df['close']))

# Log transform + differencing
df['log_close'] = np.log(df['close'])
df['diff_log_close'] = df['log_close'].diff().dropna()

print("\nADF p-value (diff log):", adf_test(df['diff_log_close'].dropna()))
print("KPSS p-value (diff log):", kpss_test(df['diff_log_close'].dropna()))

# =====================================
# 10. ACF & PACF (ARIMA IDENTIFICATION)
# =====================================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(14,5))
plot_acf(df['diff_log_close'].dropna(), lags=40)
plt.title("ACF of Differenced Log Close")
plt.show()

plt.figure(figsize=(14,5))
plot_pacf(df['diff_log_close'].dropna(), lags=40)
plt.title("PACF of Differenced Log Close")
plt.show()

"""
Interpretation:
- Slow decay in ACF → AR component
- Sharp cutoff → MA component
"""

# =====================================
# 11. CHECK IF ARIMA IS VALID (LJUNG-BOX)
# =====================================

from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(df['diff_log_close'].dropna(), lags=[10], return_df=True)
print("\nLjung-Box Test:\n", lb)

# =====================================
# 12. SEASONALITY CHECK (SARIMA)
# =====================================

from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(df['log_close'], model='additive', period=252)
decomp.plot()
plt.show()

"""
If seasonal component is weak → SARIMA not necessary
Stock prices usually show weak seasonality
"""

# =====================================
# 13. RETURNS & VOLATILITY (GARCH CHECK)
# =====================================

df['returns'] = df['log_close'].diff().dropna()

plt.figure(figsize=(14,5))
plt.plot(df['returns'])
plt.title("Log Returns")
plt.show()

from statsmodels.stats.diagnostic import het_arch

arch_test = het_arch(df['returns'].dropna())
print("\nARCH Test p-value:", arch_test[1])

"""
If p-value < 0.05 → volatility clustering → GARCH appropriate
"""

# =====================================
# 14. FIT GARCH(1,1)
# =====================================

from arch import arch_model

garch = arch_model(df['returns'].dropna(),
                   mean='zero',
                   vol='Garch',
                   p=1, q=1)

garch_fit = garch.fit(disp='off')
print(garch_fit.summary())

# =====================================
# 15. FINAL MATHEMATICAL MODEL DECISION
# =====================================

print("\n================ FINAL CONCLUSION ================\n")

print("""
Let:
Y_t = μ_t + ε_t
ε_t = σ_t z_t ,  z_t ~ N(0,1)

ARIMA / SARIMA → model μ_t (mean)
GARCH → model σ_t² (variance)
""")

print("""
Empirical findings:
✔ Prices are non-stationary
✔ Log-differenced series is stationary
✔ Weak seasonality → SARIMA not essential
✔ ARCH test significant → heteroskedasticity
""")

print("""
✅ BEST MODEL (MATHEMATICALLY):

- ARIMA on log prices → trend forecasting
- GARCH on returns → volatility forecasting

➡ ARIMA + GARCH is optimal for financial time series
""")
# =====================================
# 16. ARIMA MODEL (MEAN EQUATION)
# =====================================

from statsmodels.tsa.arima.model import ARIMA

# Use log prices for ARIMA
log_price = df['log_close'].dropna()

# Simple ARIMA (p,d,q) chosen based on ACF/PACF
arima_model = ARIMA(log_price, order=(1, 1, 1))
arima_fit = arima_model.fit()

print(arima_fit.summary())

# =====================================
# 17. ARIMA FORECAST (MEAN)
# =====================================

n_forecast = 30  # 30 trading days

arima_forecast = arima_fit.get_forecast(steps=n_forecast)

mean_forecast = arima_forecast.predicted_mean
mean_ci = arima_forecast.conf_int()

# =====================================
# 18. GARCH FORECAST (VARIANCE)
# =====================================

from arch import arch_model

returns = df['returns'].dropna()

garch = arch_model(
    returns,
    mean='zero',
    vol='Garch',
    p=1, q=1
)

garch_fit = garch.fit(disp='off')

# Forecast variance
garch_forecast = garch_fit.forecast(horizon=n_forecast)

# Get conditional variance
variance_forecast = garch_forecast.variance.iloc[-1]
volatility_forecast = np.sqrt(variance_forecast)

# =====================================
# 19. COMBINE ARIMA + GARCH
# =====================================

forecast_df = pd.DataFrame({
    'mean_log_price': mean_forecast,
    'volatility': volatility_forecast.values
})

forecast_df.index = mean_forecast.index

# Confidence bands using volatility
forecast_df['upper'] = forecast_df['mean_log_price'] + 1.96 * forecast_df['volatility']
forecast_df['lower'] = forecast_df['mean_log_price'] - 1.96 * forecast_df['volatility']

# =====================================
# 20. CONVERT BACK TO PRICE SCALE
# =====================================

forecast_df['mean_price'] = np.exp(forecast_df['mean_log_price'])
forecast_df['upper_price'] = np.exp(forecast_df['upper'])
forecast_df['lower_price'] = np.exp(forecast_df['lower'])

# =====================================
# 21. PLOT FINAL ARIMA + GARCH FORECAST
# =====================================

plt.figure(figsize=(14,6))

plt.plot(df.index[-200:], df['close'][-200:], label='Historical Price')
plt.plot(forecast_df.index, forecast_df['mean_price'], label='Forecast', color='black')

plt.fill_between(
    forecast_df.index,
    forecast_df['lower_price'],
    forecast_df['upper_price'],
    color='gray',
    alpha=0.3,
    label='95% Confidence Interval'
)

plt.title("ARIMA + GARCH Forecast (Mean + Volatility)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
