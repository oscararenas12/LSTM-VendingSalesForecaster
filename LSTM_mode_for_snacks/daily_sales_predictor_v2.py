# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import os
import math

# %% Load snack sales data
df = pd.read_csv('../data/cleaned_snacks_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['TSLP'] = df['Date'].diff().dt.total_seconds().fillna(0)

# Daily aggregation
daily = df.groupby(df['Date'].dt.date).agg(
    Daily_Items_Sold=('# of Items', 'sum'),
    Avg_TSLP=('TSLP', 'mean')
).reset_index()

# Feature engineering
daily['Date'] = pd.to_datetime(daily['Date'])
daily['Day_of_Week'] = daily['Date'].dt.dayofweek
daily['Day_cos'] = np.cos(2 * np.pi * daily['Day_of_Week'] / 7)
daily['Lagged_Items_1'] = daily['Daily_Items_Sold'].shift(1)
daily = daily.dropna()

features = ['Day_cos', 'Lagged_Items_1', 'Avg_TSLP']
target = 'Daily_Items_Sold'

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(daily[features + [target]])
scaled_df = pd.DataFrame(scaled, columns=features + [target])

# Create sequences
sequence_len = 1
X, y = [], []
for i in range(len(scaled_df) - sequence_len):
    X.append(scaled_df.iloc[i:i+sequence_len][features].values)
    y.append(scaled_df.iloc[i+sequence_len][target])
X, y = np.array(X), np.array(y)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Predict
predicted = model.predict(X_test).flatten()
X_test_last = X_test[:, -1, :]
actual_rescaled = scaler.inverse_transform(np.hstack([X_test_last, y_test.reshape(-1, 1)]))[:, -1]
predicted_rescaled = scaler.inverse_transform(np.hstack([X_test_last, predicted.reshape(-1, 1)]))[:, -1]

real_rmse = math.sqrt(mean_squared_error(actual_rescaled, predicted_rescaled))
print(f"\n📉 Snack RMSE (v2 - LSTM): {real_rmse:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual_rescaled, label='Actual')
plt.plot(predicted_rescaled, label='Predicted')
plt.title("Snack Sales Forecast - v2 (LSTM)")
plt.xlabel("Time (Days)")
plt.ylabel("Items Sold")
plt.legend()
plt.tight_layout()
plt.show()
# %%
