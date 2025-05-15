#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
register_matplotlib_converters()

#%% Load Data
df = pd.read_csv('../../data/cleaned_drink_sales_data.csv')

#%% Preprocess Data
df['Date'] = pd.to_datetime(df['Date'])
df['# of Items'] = pd.to_numeric(df['# of Items'], errors='coerce')
df.set_index('Date', inplace=True)

# Resample by day
df_daily = df.resample('D')['# of Items'].sum().to_frame()
df_daily.rename(columns={'# of Items': 'total_items_sold'}, inplace=True)

#%% Plot daily total items sold
plt.figure(figsize=(12, 4))
plt.plot(df_daily['total_items_sold'], label='Total Items Sold')
plt.title('Drink Vending Machine Sales Over Time', fontsize=20)
plt.ylabel('Total Items Sold', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%% ACF and PACF plots
series = df_daily['total_items_sold'].dropna()

plt.figure(figsize=(10, 4))
plot_acf(series, lags=30)
plt.title('ACF: Daily Total Items Sold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(series, lags=15, method='ywm')
plt.title('PACF: Daily Total Items Sold')
plt.tight_layout()
plt.show()