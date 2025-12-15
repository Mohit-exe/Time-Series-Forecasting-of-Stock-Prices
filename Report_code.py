#!/usr/bin/env python
# coding: utf-8

# # Time Series Forecasting of NFLX Stock Prices
# 
# This project performs exploratory data analysis, stationarity testing,
# ARIMA modeling, GARCH volatility modeling, and combined ARIMA+GARCH forecasting
# on Netflix stock prices.
# 

# In[17]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')



# In[2]:


file_path = r"C:/Users/USER/Music/Self Projects/Time Series forecasting of Stock Prices/NFLX_stocks.csv"
df = pd.read_csv(file_path)

if 'unnamed: 0' in df.columns:
    df.rename(columns={'unnamed: 0': 'date'}, inplace=True)

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

df['close'] = pd.to_numeric(df['close'], errors='coerce')
df = df.dropna(subset=['close'])

df.head()


# In[3]:


df['close'].describe()


# In[4]:


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


# In[5]:


monthly_close = df['close'].resample('M').mean()

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


# In[6]:


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


# In[7]:


rolling_mean = df['close'].rolling(window=252).mean()
rolling_std = df['close'].rolling(window=252).std()

plt.figure(figsize=(14,6))
plt.plot(df['close'], label='Close Price', alpha=0.6)
plt.plot(rolling_mean, label='252-Day Rolling Mean', linewidth=2)
plt.plot(rolling_std, label='252-Day Rolling Std', linewidth=2)

plt.title("NFLX Rolling Mean & Volatility")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[8]:


from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series):
    return adfuller(series, autolag='AIC')[1]

def kpss_test(series):
    return kpss(series, regression='c', nlags="auto")[1]

print("ADF (level):", adf_test(df['close']))
print("KPSS (level):", kpss_test(df['close']))

df['log_close'] = np.log(df['close'])
df['diff_log_close'] = df['log_close'].diff()

print("ADF (diff log):", adf_test(df['diff_log_close'].dropna()))
print("KPSS (diff log):", kpss_test(df['diff_log_close'].dropna()))


# In[9]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['diff_log_close'].dropna(), lags=40)
plt.show()

plot_pacf(df['diff_log_close'].dropna(), lags=40)
plt.show()


# In[10]:


from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(df['diff_log_close'].dropna(), lags=[10], return_df=True)


# In[11]:


from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(df['log_close'], model='additive', period=252).plot()
plt.show()


# In[12]:


df['returns'] = 100 * df['log_close'].diff()

plt.figure(figsize=(14,5))
plt.plot(df['returns'])
plt.title("Log Returns")
plt.show()

from statsmodels.stats.diagnostic import het_arch
het_arch(df['returns'].dropna())


# In[13]:


from arch import arch_model

garch = arch_model(df['returns'].dropna(),
                   mean='zero',
                   vol='Garch',
                   p=1, q=1)

garch_fit = garch.fit(disp='off')
print(garch_fit.summary())


# In[14]:


from statsmodels.tsa.arima.model import ARIMA

arima = ARIMA(df['log_close'].dropna(), order=(1,1,1))
arima_fit = arima.fit()
print(arima_fit.summary())


# In[15]:


# ================================
# FIXED ARIMA + GARCH FORECAST
# ================================

n_forecast = 30

# ---------- ARIMA forecast ----------
arima_forecast = arima_fit.get_forecast(steps=n_forecast)
mean_log_forecast = arima_forecast.predicted_mean

# Create proper future date index
last_date = df.index[-1]
forecast_index = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=n_forecast,
    freq='B'  # business days
)

mean_log_forecast.index = forecast_index

# ---------- GARCH forecast ----------
garch_forecast = garch_fit.forecast(horizon=n_forecast)

# Volatility of RETURNS
volatility = np.sqrt(garch_forecast.variance.iloc[-1].values)

# Scale volatility correctly (returns are %)
volatility = volatility / 100  

# ---------- Combine correctly ----------
forecast_df = pd.DataFrame(index=forecast_index)
forecast_df['mean_log'] = mean_log_forecast.values
forecast_df['upper_log'] = forecast_df['mean_log'] + 1.96 * volatility
forecast_df['lower_log'] = forecast_df['mean_log'] - 1.96 * volatility

# Convert to price
forecast_df['mean_price'] = np.exp(forecast_df['mean_log'])
forecast_df['upper_price'] = np.exp(forecast_df['upper_log'])
forecast_df['lower_price'] = np.exp(forecast_df['lower_log'])


# In[16]:


plt.figure(figsize=(14,6))

# Plot recent history only (important)
plt.plot(df.index[-300:], df['close'][-300:], label='Historical', linewidth=2)

# Plot forecast
plt.plot(forecast_df.index, forecast_df['mean_price'],
         label='Forecast', linewidth=2, color='black')

# Confidence bands
plt.fill_between(
    forecast_df.index,
    forecast_df['lower_price'],
    forecast_df['upper_price'],
    alpha=0.3,
    label='95% Confidence Interval'
)

plt.title("ARIMA + GARCH Forecast (Corrected)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ## Conclusion
# 
# Stock prices are non-stationary, while returns are stationary with strong
# volatility clustering. ARIMA effectively models the conditional mean,
# and GARCH captures time-varying volatility, making the combined
# ARIMA + GARCH framework optimal for financial time series forecasting.
# 

# In[ ]:




