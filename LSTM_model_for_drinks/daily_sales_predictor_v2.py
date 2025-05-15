# %% 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# %% 📁 Load and preprocess the dataset
df = pd.read_csv("../../data/cleaned_drink_sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Date_Only'] = df['Date'].dt.date
df['Hour_of_Day'] = df['Date'].dt.hour

# Define peak hour range (15:00–20:00 inclusive)
peak_hours = list(range(15, 21))
df['Is_Peak_Hour'] = df['Hour_of_Day'].isin(peak_hours).astype(int)

# Aggregate to daily level
daily = df.groupby('Date_Only').agg(
    Total_Items_Sold=('# of Items', 'sum'),
    Total_Revenue=('Amount', 'sum'),
    Is_Peak_Hour=('Is_Peak_Hour', 'max')  # 1 if any sale in peak hour
).reset_index()

daily['Date'] = pd.to_datetime(daily['Date_Only'])
daily['Day_of_Week'] = daily['Date'].dt.dayofweek
daily['Is_Weekend'] = daily['Day_of_Week'].isin([5, 6]).astype(int)

# Keep only relevant columns
data = daily[['Date', 'Total_Items_Sold', 'Day_of_Week', 'Is_Weekend', 'Is_Peak_Hour']].copy()
print(data)

# %% 📏 Normalize the features
scalers = {}
for col in ['Total_Items_Sold', 'Day_of_Week', 'Is_Weekend', 'Is_Peak_Hour']:
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

SEQ_LEN = 7
feature_data = data[['Total_Items_Sold', 'Day_of_Week', 'Is_Weekend', 'Is_Peak_Hour']].values
X, y = create_sequences(feature_data, SEQ_LEN)

# %% 🧪 Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# %% 🧠 Build and compile the LSTM model
# Set your desired save directory
save_dir = r"C:\Users\oscar\Documents\VS\salesForecasting\LSTM-oscar\model_5_11"
os.makedirs(save_dir, exist_ok=True)  # Ensure it exists

# Define learning rates to try
learning_rates = [0.01, 0.001, 0.0005, 0.0001]
results = []

for lr in learning_rates:
    print(f"\n🚀 Training with learning rate: {lr}")

    # Rebuild model
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQ_LEN, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')

    # Create full checkpoint path
    checkpoint_filename = f"checkpoint_lr{str(lr).replace('.', '_')}.weights.h5"
    checkpoint_filepath = os.path.join(save_dir, checkpoint_filename)

    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=16,
        callbacks=[model_checkpoint],
        verbose=0
    )

    # Record best result
    min_val_loss = min(history.history['val_loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1
    results.append((lr, min_val_loss, checkpoint_filepath))
    print(f"✅ LR={lr} → Best val_loss: {min_val_loss:.6f} at epoch {best_epoch} | saved: {checkpoint_filepath}")

# Summary
print("\n📊 Learning Rate Results:")
for lr, val_loss, filepath in results:
    print(f" - LR {lr:<8} → val_loss: {val_loss:.6f} | weights: {filepath}")

# Best LR
best_lr, best_loss, best_file = min(results, key=lambda x: x[1])
print(f"\n🏆 Best learning rate: {best_lr} with val_loss: {best_loss:.6f}")
print(f"📦 Best model weights saved to: {best_file}")

#%%
# Load best model weights
model.load_weights(best_file)

# %% 🔮 Predict and inverse transform
y_pred = model.predict(X_test)
y_pred_inv = scalers['Total_Items_Sold'].inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scalers['Total_Items_Sold'].inverse_transform(y_test.reshape(-1, 1))

# %% 📊 Plot predictions vs actual
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Combined LSTM Forecast: Items Sold (with Is_Peak_Hour)')
plt.xlabel('Time Step')
plt.ylabel('Items Sold')
plt.legend()
plt.grid()
plt.show()
# %%

import seaborn as sns
import matplotlib.pyplot as plt

# If not already numeric
data_corr = data.drop(columns=['Date'])

# Compute Pearson correlation matrix
corr = data_corr.corr()

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# %%
# %% 📊 Evaluate Model Performance
from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(y_test_inv, y_pred_inv)

# Calculate RMSE
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Optional: Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(y_test_inv - y_pred_inv))
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Optional: Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# %% 📊 Plot predictions vs actual with error metrics in title
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title(f'Combined LSTM Forecast: Items Sold (RMSE: {rmse:.2f}, MSE: {mse:.2f})')
plt.xlabel('Time Step')
plt.ylabel('Items Sold')
plt.legend()
plt.grid()
plt.show()