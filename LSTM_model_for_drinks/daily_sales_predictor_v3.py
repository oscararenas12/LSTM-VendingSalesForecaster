#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

#%%
# Load and preprocess data
df = pd.read_csv('../../data/cleaned_drink_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['TSLP'] = df['Date'].diff().dt.total_seconds().fillna(0)
df['Is_Cash'] = (df['Card Type'] == 'Cash').astype(int)
df['Is_Card'] = (df['Is_Cash'] == 0).astype(int)

# Aggregate daily
daily = df.groupby(df['Date'].dt.date).agg(
    Daily_Items_Sold=('# of Items', 'sum'),
    Card_Transactions=('Is_Card', 'sum'),
    Avg_TSLP=('TSLP', 'mean')
).reset_index()

# Add cyclical & lag features (no week sin/cos here)
daily['Date'] = pd.to_datetime(daily['Date'])
daily['Day_of_Week'] = daily['Date'].dt.dayofweek
daily['Day_sin'] = np.sin(2 * np.pi * daily['Day_of_Week'] / 7)
daily['Day_cos'] = np.cos(2 * np.pi * daily['Day_of_Week'] / 7)
daily['Lagged_Items_1'] = daily['Daily_Items_Sold'].shift(1)
daily['Rolling_3_Day_Sales'] = daily['Daily_Items_Sold'].rolling(window=3).mean()
daily = daily.dropna()

# Select features and target
features = ['Card_Transactions', 'Rolling_3_Day_Sales', 'Lagged_Items_1', 'Avg_TSLP',
            'Day_sin', 'Day_cos']
target = 'Daily_Items_Sold'

#%%
# Normalize with MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(daily[features + [target]])
scaled_df = pd.DataFrame(scaled, columns=features + [target])

# Create 7-day sequences
sequence_len = 1
X, y = [], []
for i in range(len(scaled_df) - sequence_len):
    X.append(scaled_df.iloc[i:i+sequence_len][features].values)
    y.append(scaled_df.iloc[i+sequence_len][target])
X, y = np.array(X), np.array(y)

#%%
# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#%%
# Build LSTM model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True,
                       kernel_initializer=initializers.glorot_uniform(),
                       recurrent_initializer=initializers.orthogonal()),
                  input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False,
         kernel_initializer=initializers.glorot_uniform(),
         recurrent_initializer=initializers.orthogonal()),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=1)

#%%
# Predict (normalized)
predicted = model.predict(X_test).flatten()

# Inverse transform for real-scale MSE
X_test_last = X_test[:, -1, :]
actual_rescaled = scaler.inverse_transform(np.hstack([X_test_last, y_test.reshape(-1, 1)]))[:, -1]
predicted_rescaled = scaler.inverse_transform(np.hstack([X_test_last, predicted.reshape(-1, 1)]))[:, -1]

#%%
#%%
# Compute MSE, RMSE, MAE, and MAPE
from sklearn.metrics import mean_squared_error, mean_absolute_error

real_mse = mean_squared_error(actual_rescaled, predicted_rescaled)
real_rmse = math.sqrt(real_mse)
real_mae = mean_absolute_error(actual_rescaled, predicted_rescaled)

# Calculate MAPE manually (avoiding division by zero)
epsilon = 1e-10  # Small value to avoid division by zero
real_mape = np.mean(np.abs((actual_rescaled - predicted_rescaled) / (actual_rescaled + epsilon))) * 100

print(f"📊 Real-Scale MSE: {real_mse:.2f}")
print(f"📉 Real-Scale RMSE: {real_rmse:.2f}")
print(f"📏 Real-Scale MAE: {real_mae:.2f}")
print(f"📈 Real-Scale MAPE: {real_mape:.2f}%")

# Plot results with metrics in title
plt.figure(figsize=(12, 6))
plt.plot(actual_rescaled, label='Actual')
plt.plot(predicted_rescaled, label='Predicted')
plt.title(f"Bidirectional LSTM: Daily Items Sold (RMSE: {real_rmse:.2f}, MAE: {real_mae:.2f}, MAPE: {real_mape:.2f}%)")
plt.xlabel("Time (Days)")
plt.ylabel("Items Sold")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Create residual plot to visualize errors
plt.figure(figsize=(12, 4))
residuals = actual_rescaled - predicted_rescaled
plt.scatter(range(len(residuals)), residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title("Prediction Residuals (Errors)")
plt.xlabel("Time (Days)")
plt.ylabel("Error (Actual - Predicted)")
plt.tight_layout()
plt.show()
# %%
