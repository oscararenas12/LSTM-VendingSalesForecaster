# %% 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% 📁 Load and preprocess base data
df = pd.read_csv('../data/cleaned_drink_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

#%%
# Base features
df['Hour_of_Day'] = df['Date'].dt.hour
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
df['Hour_Bin'] = pd.cut(df['Hour_of_Day'], bins=[-1, 5, 11, 17, 23],
                        labels=['Early_Morning', 'Morning', 'Afternoon', 'Evening'])
df = df.sort_values('Date')
df['TSLP'] = df['Date'].diff().dt.total_seconds().fillna(0)
df['Is_Cash'] = (df['Card Type'] == 'Cash').astype(int)
df['Is_Contactless'] = df['Card Type'].str.contains('Contactless|Apple Pay', case=False, na=False).astype(int)
df['Is_Card'] = (df['Is_Cash'] == 0).astype(int)
df['Rounded_Amount'] = (df['Amount'] * 2).round() / 2

# %% 📊 Daily aggregation
daily = df.groupby(df['Date'].dt.date).agg(
    Daily_Revenue=('Amount', 'sum'),
    Daily_Items_Sold=('# of Items', 'sum'),
    Daily_Transactions=('Tran #', 'count'),
    Avg_TSLP=('TSLP', 'mean'),
    Cash_Transactions=('Is_Cash', 'sum'),
    Card_Transactions=('Is_Card', 'sum'),
    Avg_Amount=('Amount', 'mean')
).reset_index()

# Convert daily['Date'] to datetime64[ns]
daily['Date'] = pd.to_datetime(daily['Date'])

# Add lag, rolling, ratio, and cyclical features
daily['Lagged_Items_1'] = daily['Daily_Items_Sold'].shift(1)
daily['Lagged_Items_2'] = daily['Daily_Items_Sold'].shift(2)
daily['Lagged_Items_3'] = daily['Daily_Items_Sold'].shift(3)
daily['Lagged_Revenue_1'] = daily['Daily_Revenue'].shift(1)
daily['Rolling_3_Day_Sales'] = daily['Daily_Items_Sold'].rolling(window=3).mean()
daily['Rolling_7_Day_Sales'] = daily['Daily_Items_Sold'].rolling(window=7).mean()
daily['Txn_per_Item'] = daily['Daily_Transactions'] / daily['Daily_Items_Sold']
daily['Pct_Cash'] = daily['Cash_Transactions'] / daily['Daily_Transactions']
daily['Pct_Card'] = daily['Card_Transactions'] / daily['Daily_Transactions']
daily['Day_of_Week'] = daily['Date'].dt.dayofweek

# Add average DOW sales
dow_avg = daily.groupby('Day_of_Week')['Daily_Items_Sold'].mean()
daily['DOW_Sales_Avg'] = daily['Day_of_Week'].map(dow_avg)

# %% 🕒 Hour bin-based daily sales
hourly = df.groupby([df['Date'].dt.date, 'Hour_Bin']).agg(
    Hourly_Sales=('# of Items', 'sum')
).reset_index()

hour_pivot = hourly.pivot(index='Date', columns='Hour_Bin', values='Hourly_Sales').fillna(0).reset_index()
hour_pivot.columns.name = None

# 🔧 FIX: Ensure matching datetime64[ns] type
hour_pivot['Date'] = pd.to_datetime(hour_pivot['Date'])

# Merge into daily
daily = pd.merge(daily, hour_pivot, on='Date', how='left')

# Drop NA rows for clean correlation
daily_clean = daily.dropna()

# %% 🔍 Plot correlation heatmap
corr_matrix = daily_clean.corr(numeric_only=True)

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("📊 Expanded Correlation Heatmap (Daily Level Features)")
plt.tight_layout()
plt.show()

# %% 🔍 Focused Correlation Heatmap (Strong/Moderate Only)
threshold = 0.3  # Minimum correlation value (absolute) to include in heatmap

# Compute correlation matrix
corr_matrix = daily_clean.corr(numeric_only=True)

# Focus on correlations with Daily_Items_Sold
target_corr = corr_matrix['Daily_Items_Sold'].abs()
relevant_features = target_corr[target_corr >= threshold].index.tolist()

# Slice the matrix
focused_corr = corr_matrix.loc[relevant_features, relevant_features]

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(focused_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("📊 Focused Correlation Heatmap (Filtered by |r| ≥ 0.3)")
plt.tight_layout()
plt.show()
