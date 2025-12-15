# Time Series Forecasting of NFLX Stock Prices using ARIMA–GARCH

## 📌 Project Overview
This project performs an end-to-end time series analysis and forecasting of **Netflix (NFLX) stock prices** using classical econometric models.  
The objective is to model:

- **Trend / Mean behavior** using **ARIMA**
- **Time-varying volatility** using **GARCH**
- Combine both into a **risk-aware forecast**

The analysis follows a rigorous statistical workflow including exploratory data analysis, stationarity testing, model diagnostics, and forecasting.

---

## 📂 Dataset
- **File**: `NFLX_stocks.csv`
- **Frequency**: Daily
- **Key Columns**:
  - `date` – trading date
  - `close` – closing price

> All column names are assumed to be in **lower case**.

---

## 🛠️ Technologies Used
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- ARCH

---

## 🔍 Methodology

### 1️⃣ Data Cleaning & Preprocessing
- Converted date column to `datetime`
- Set date as index
- Converted price data to numeric
- Removed missing or invalid values

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Time series plot of closing prices
- Monthly resampled visualization
- Log-scale visualization
- Rolling mean and rolling volatility (252 trading days)

---

### 3️⃣ Stationarity Analysis
- **ADF Test** (Null: non-stationary)
- **KPSS Test** (Null: stationary)
- Log transformation and first differencing applied

✔ Prices are non-stationary  
✔ Log-differenced series is stationary

---

### 4️⃣ ARIMA Model (Mean Equation)
- ACF and PACF used for order identification
- ARIMA fitted on **log prices**
- Ljung–Box test used for residual diagnostics

ARIMA models the **conditional mean** of the time series.

---

### 5️⃣ Seasonality Check
- Seasonal decomposition with yearly period (252 trading days)
- Weak seasonality observed

➡ SARIMA not required

---

### 6️⃣ Volatility Modeling (GARCH)
- Log returns computed
- ARCH test confirms heteroskedasticity
- GARCH(1,1) fitted to returns

GARCH captures **volatility clustering**, common in financial data.

---

### 7️⃣ ARIMA + GARCH Forecasting
- ARIMA forecasts the **mean log-price**
- GARCH forecasts **conditional volatility**
- Combined to generate:
  - Point forecasts
  - Volatility-adjusted confidence intervals

The final output provides both **expected price path** and **risk bounds**.

---

## 📈 Results
- ARIMA alone produces a flat multi-step mean forecast (expected behavior)
- GARCH widens uncertainty bands based on forecasted volatility
- The combined ARIMA–GARCH framework is mathematically appropriate for financial time series

---

## 📊 Final Conclusion
Stock prices are non-stationary, while returns are stationary with significant volatility clustering.  
Therefore:

> **ARIMA is suitable for modeling the conditional mean, and GARCH is optimal for modeling conditional variance.**

The **ARIMA + GARCH** framework provides the most robust approach for financial time series forecasting.

---

## ▶️ How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels arch
