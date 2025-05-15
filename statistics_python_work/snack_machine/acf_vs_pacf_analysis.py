#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
register_matplotlib_converters()

# %% Load Data
df = pd.read_csv('../../data/cleaned_snack_sales_data.csv')

# %%
df['Date'] = pd.to_datetime(df['Date'])

# Convert '# of Items' to numeric (some might be strings)
df['# of Items'] = pd.to_numeric(df['# of Items'], errors='coerce')

# Set 'Date' as index
df.set_index('Date', inplace=True)

# Resample by day, summing the total items sold
df_daily = df.resample('D')['# of Items'].sum().to_frame()

# Rename for clarity
df_daily.rename(columns={'# of Items': 'total_items_sold'}, inplace=True)

# Show result
print(df_daily)

# %%
plt.figure(figsize=(12, 4))
plt.plot(df_daily['total_items_sold'], label='Total Items Sold')

plt.title('Snack Vending Machine Sales Over Time', fontsize=20)
plt.ylabel('Total Items Sold', fontsize=16)
plt.xlabel('Date', fontsize=16)

# Add vertical lines at the start of each month
start = df_daily.index.min().replace(day=1)
end = df_daily.index.max().replace(day=1)
month_range = pd.date_range(start, end, freq='MS')  # 'MS' = month start

for month in month_range:
    plt.axvline(month, color='gray', linestyle='--', alpha=0.2)

plt.tight_layout()
plt.grid(alpha=0.3)
plt.legend()
plt.show()


# %%
# Drop NA if any
series = df_daily['total_items_sold'].dropna()

# ACF plot (up to 30 days lag)
plt.figure(figsize=(10, 4))
plot_acf(series, lags=30)
plt.title('ACF: Daily Total Items Sold')
plt.tight_layout()
plt.show()

# PACF plot (up to 15 days lag)
plt.figure(figsize=(10, 4))
plot_pacf(series, lags=15, method='ywm')  # Yule-Walker method
plt.title('PACF: Daily Total Items Sold')
plt.tight_layout()
plt.show()

# %% 🔍 ADF Test on Original Series
series = df_daily['total_items_sold'].dropna()
adf_result_raw = adfuller(series)

print("=== ADF Test on Original Series ===")
print("ADF Statistic:", adf_result_raw[0])
print("p-value:", adf_result_raw[1])
for key, value in adf_result_raw[4].items():
    print(f"Critical Value ({key}): {value}")
print()

#%%
plt.plot(df_daily['total_items_sold'])
plt.title("Raw Daily Sales")
plt.grid(True)
plt.show()

# %% ACF/PACF for Original Series (stationary confirmed)
plt.figure(figsize=(10, 4))
plot_acf(series, lags=30)
plt.title("ACF of Original Series (Stationary)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(series, lags=15, method='ywm')
plt.title("PACF of Original Series (Stationary)")
plt.tight_layout()
plt.show()

# %% 🧪 ADF Test on Differenced Series
df_diff = series.diff().dropna() # 🔁 First Differencing to Make Series Stationary

adf_result = adfuller(df_diff)

print("=== ADF Test on Differenced Series ===")
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value}")
print()

# %% 📉 ACF & PACF Plots of Differenced Series
plt.figure(figsize=(10, 4))
plot_acf(df_diff, lags=30)
plt.title("ACF of First-Differenced Series")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(df_diff, lags=15, method='ywm')
plt.title("PACF of First-Differenced Series")
plt.tight_layout()
plt.show()

# %% 🤖 Fit ARIMA(1, 0, 2) and evaluate AIC/BIC
from statsmodels.tsa.arima.model import ARIMA

# Fit the model to the original (stationary) series
model = ARIMA(series, order=(1, 0, 2))  # ARIMA(p=1, d=0, q=2)
model_fit = model.fit()

# Print model summary including AIC and BIC
print(model_fit.summary())

# Plot fitted vs. actual
plt.figure(figsize=(12, 5))
plt.plot(series, label="Actual")
plt.plot(model_fit.fittedvalues, label="Fitted", linestyle='--')
plt.title("ARIMA(1, 0, 2): Actual vs Fitted")
plt.xlabel("Date")
plt.ylabel("Total Items Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% 🔮 Predict the next day's sales
# Fit ARIMA(1, 0, 2) on full series
model = ARIMA(series, order=(1, 0, 2))
model_fit = model.fit()

# Predict just the next day
next_date = series.index[-1] + pd.Timedelta(days=1)
forecast = model_fit.predict(start=next_date, end=next_date)

print(f"📅 Predicted sales for {next_date.date()}: {forecast.iloc[0]:.2f}")

# Optional: Plot the last few days + next day forecast
plt.figure(figsize=(10, 4))
plt.plot(series[-14:], label='Recent Sales')
plt.axvline(next_date, color='gray', linestyle='--', alpha=0.5)
plt.scatter(next_date, forecast.iloc[0], color='red', label='Next Day Forecast')
plt.title("Next Day Forecast using ARIMA(1, 0, 2)")
plt.xlabel("Date")
plt.ylabel("Total Items Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%