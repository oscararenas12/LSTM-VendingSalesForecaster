# V3 Model looking into 'Card_Transactions', 'Rolling_3_Day_Sales', 'Lagged_Items_1', 'Avg_TSLP', 'Day_sin', 'Day_cos'

# %% 📦 Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

#%% # Load and preprocess data
df = pd.read_csv('../data/cleaned_drink_sales_data.csv')
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

#%% # Normalize with MinMaxScaler
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

#%% # Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#%% # Build LSTM model
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

#%% # Predict (normalized)
predicted = model.predict(X_test).flatten()

# Inverse transform for real-scale MSE
X_test_last = X_test[:, -1, :]
actual_rescaled = scaler.inverse_transform(np.hstack([X_test_last, y_test.reshape(-1, 1)]))[:, -1]
predicted_rescaled = scaler.inverse_transform(np.hstack([X_test_last, predicted.reshape(-1, 1)]))[:, -1]

# %% 📐 Error Metrics
avg_daily_items_sold = df['# of Items'].sum() / df['Date'].dt.date.nunique()
print("Average Daily Items Sold:", avg_daily_items_sold)

mse = mean_squared_error(actual_rescaled, predicted_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_rescaled, predicted_rescaled)

# Robust MAPE calculation with near-zero filter
epsilon = 1e-3
mask = actual_rescaled > epsilon
if np.any(mask):
    mape = np.mean(np.abs((actual_rescaled[mask] - predicted_rescaled[mask]) / actual_rescaled[mask])) * 100
else:
    mape = np.nan

print(f"📊 Mean Squared Error (MSE): {mse:.2f}")
print(f"📉 Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"📏 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 📊 Enhanced plot with error metrics
plt.figure(figsize=(12, 5))
plt.plot(actual_rescaled, label='Actual')
plt.plot(predicted_rescaled, label='Predicted')
plt.title(f'Bidirectional LSTM Forecast: Items Sold (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%)')
plt.xlabel("Time Step")
plt.ylabel("Items Sold")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('../images/Combined_LSTM_Forecast_V3.png', dpi=300, bbox_inches='tight')
plt.show()

# 📊 Error distribution visualization
plt.figure(figsize=(12, 5))
errors = actual_rescaled - predicted_rescaled

# Histogram of errors
plt.subplot(1, 2, 1)
plt.hist(errors, bins=20, alpha=0.7)
plt.axvline(0, color='r', linestyle='--')
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')

# Scatter plot of actual vs predicted
plt.subplot(1, 2, 2)
plt.scatter(actual_rescaled, predicted_rescaled, alpha=0.5)
plt.plot([actual_rescaled.min(), actual_rescaled.max()],
         [actual_rescaled.min(), actual_rescaled.max()], 'r--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.savefig('../images/Error_Distribution_and_Actual_vs_Predicted_V3.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
