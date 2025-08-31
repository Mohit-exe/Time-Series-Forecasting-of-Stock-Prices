# 📈 Stock Price Time Series Analysis

This project performs **time series analysis** on stock market data using Python.  
The main objective is to explore stock price trends, test for stationarity, and apply transformations such as differencing and smoothing.

---

## 📂 Project Overview
The steps followed in this project:
1. Load and clean stock data from CSV.
2. Visualize the stock’s **High Price** over time.
3. Perform **monthly resampling** to analyze long-term trends.
4. Plot the **Autocorrelation Function (ACF)** to detect serial correlations.
5. Conduct **Augmented Dickey-Fuller (ADF) test** for stationarity.
6. Apply **first differencing** to remove trends and stabilize variance.
7. Use **rolling averages** for smoothing and trend visualization.

---

## 📊 Dataset
- **File Used:** `stock_data.csv`
- **Columns:**
  - `Date` (datetime index)
  - `High` (highest stock price per day)
  - `Volume` (trading volume)
  - Other stock-related columns may also be present.

---

## ⚙️ Technologies Used
- Python 🐍
- Pandas (data manipulation & resampling)
- NumPy
- Matplotlib & Seaborn (visualization)
- Statsmodels (ACF plot & ADF stationarity test)

---

## 📈 Analysis & Visualizations
- **Line Plot:** Daily High Prices over time.
- **Monthly Resampling:** Average High Prices aggregated by month.
- **ACF Plot:** Autocorrelation structure of trading volume.
- **ADF Test:** Statistical test for stationarity of time series.
- **Differencing:** Original vs differenced series to remove trends.
- **Moving Average:** Smoothed series with a rolling window.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-time-series-analysis.git
   cd stock-time-series-analysis
2.Install dependencies :
```bash
   pip install -r requirements.txt
```
3.Place the dataset (stock_data.csv) in the project folder.
4.Run the script or Jupyter Notebook:
```bash
   jupyter notebook analysis.ipynb
```

 
## 📌 Future Improvements
- 🔮 Fit ARIMA / SARIMA models for forecasting.
- 📊 Add seasonality decomposition (trend, seasonality, residuals).
- 🧮 Try GARCH models for volatility analysis.
- 🤖 Apply LSTM / Deep Learning models for predictive modeling.
- 🌍 Extend the analysis to multiple stocks for portfolio-level insights.

---

