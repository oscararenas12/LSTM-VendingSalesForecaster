# V1 Model looking into 'Date', 'Total_Items_Sold', 'Day_of_Week', 'Is_Weekend'

# %% 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %% 📁 Load and preprocess the dataset
df = pd.read_csv("../data/cleaned_drink_sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Date_Only'] = df['Date'].dt.date

# Aggregate to daily level
daily = df.groupby('Date_Only').agg(
    Total_Items_Sold=('# of Items', 'sum'),
    Total_Revenue=('Amount', 'sum')
).reset_index()

daily['Date'] = pd.to_datetime(daily['Date_Only'])
daily['Day_of_Week'] = daily['Date'].dt.dayofweek
daily['Is_Weekend'] = daily['Day_of_Week'].isin([5, 6]).astype(int)

# Only keep relevant columns
data = daily[['Date', 'Total_Items_Sold', 'Day_of_Week', 'Is_Weekend']].copy()

# %% 📏 Normalize the features
scalers = {}
for col in ['Total_Items_Sold', 'Day_of_Week', 'Is_Weekend']:
    scaler = MinMaxScaler()
    data[col] = scaler.fit_transform(data[[col]])
    scalers[col] = scaler

# %% 🧩 Create sequences with multiple features
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        seq_x = data[i:i+seq_len]
        seq_y = data[i+seq_len, 0]  # predict 'Total_Items_Sold'
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

SEQ_LEN = 2
feature_data = data[['Total_Items_Sold', 'Day_of_Week', 'Is_Weekend']].values
X, y = create_sequences(feature_data, SEQ_LEN)

# %% 🧪 Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# %% 🧠 Build and compile the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(SEQ_LEN, X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#%%
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    verbose=1,
)

# %% 🔮 Predict and inverse transform
y_pred = model.predict(X_test)
y_pred_inv = scalers['Total_Items_Sold'].inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scalers['Total_Items_Sold'].inverse_transform(y_test.reshape(-1, 1))


# %% 📐 Error Metrics

print("Average Daily Items Sold:", df['# of Items'].sum() / df['Date_Only'].nunique())

# Calculate error metrics
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# Calculate MAPE with handling for zero values
epsilon = 1e-10  # Small value to avoid division by zero
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + epsilon))) * 100

print(f"📊 Mean Squared Error (MSE): {mse:.2f}")
print(f"📉 Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"📏 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 📊 Enhanced plot with error metrics
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title(f'Combined LSTM Forecast: Items Sold (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%)')
plt.xlabel('Time Step')
plt.ylabel('Items Sold')
plt.legend()
plt.grid()
plt.savefig('../images/Combined_LSTM_Forecast_V1.png', dpi=300, bbox_inches='tight')
plt.show()

# 📊 Error distribution visualization
plt.figure(figsize=(12, 5))
errors = y_test_inv.flatten() - y_pred_inv.flatten()

# Histogram of errors
plt.subplot(1, 2, 1)
plt.hist(errors, bins=20, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')

# Scatter plot of actual vs predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test_inv, y_pred_inv, alpha=0.5)
plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.savefig('../images/Error_Distribution_and_Actual_vs_Predicted_V1.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
