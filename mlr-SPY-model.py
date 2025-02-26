import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import scatter_matrix

warnings.filterwarnings("ignore")

# Load stock market data from CSV files
# Ensure the file paths are correct based on your data location
aord = pd.read_csv('../data/indice/ALLOrdinary.csv')
nikkei = pd.read_csv('../data/indice/Nikkei225.csv')
hsi = pd.read_csv('../data/indice/HSI.csv')
daxi = pd.read_csv('../data/indice/DAXI.csv')
cac40 = pd.read_csv('../data/indice/CAC40.csv')
sp500 = pd.read_csv('../data/indice/SP500.csv')
dji = pd.read_csv('../data/indice/DJI.csv')
nasdaq = pd.read_csv('../data/indice/nasdaq_composite.csv')
spy = pd.read_csv('../data/indice/SPY.csv')

# Adjusting for timezone differences & Creating a feature panel
dicepanel = pd.DataFrame(index=spy.index)
dicepanel['spy'] = spy['Open'].shift(-1) - spy['Open']  # Daily price change
dicepanel['spy_lag1'] = dicepanel['spy'].shift(1)       # Lag feature
dicepanel['sp500'] = sp500["Open"] - sp500['Open'].shift(1)
dicepanel['nasdaq'] = nasdaq['Open'] - nasdaq['Open'].shift(1)
dicepanel['dji'] = dji['Open'] - dji['Open'].shift(1)
dicepanel['cac40'] = cac40['Open'] - cac40['Open'].shift(1)
dicepanel['daxi'] = daxi['Open'] - daxi['Open'].shift(1)
dicepanel['aord'] = aord['Close'] - aord['Open']
dicepanel['hsi'] = hsi['Close'] - hsi['Open']
dicepanel['nikkei'] = nikkei['Close'] - nikkei['Open']
dicepanel['Price'] = spy['Open']  # SPY opening price

# Handle missing values
dicepanel.fillna(method='ffill', inplace=True)  # Forward fill NaNs
dicepanel.dropna(inplace=True)                 # Drop remaining NaNs

# Save processed data
dicepanel.to_csv('../data/indice/indicepanel.csv', index=False)

# Splitting into Train & Test sets
Train = dicepanel.iloc[-2000:-1000, :]
Test = dicepanel.iloc[-1000:, :]
print(f"Train Shape: {Train.shape}, Test Shape: {Test.shape}")

# Exploratory Data Analysis (EDA)
# Generate scatter matrix to observe relationships
scatter_matrix(Train, figsize=(10, 10))
plt.show()

# Compute Correlation with SPY
corr_array = Train.iloc[:, :-1].corr()['spy']
print("Correlation of Indices with SPY:\n", corr_array)

# Define and fit the Multiple Linear Regression (MLR) model
formula = 'spy ~ spy_lag1 + sp500 + nasdaq + dji + cac40 + aord + daxi + nikkei + hsi'
lm = smf.ols(formula=formula, data=Train).fit()
print(lm.summary())  # Show model statistics

# Make Predictions
Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)

# Scatter plot: Actual vs Predicted SPY movement
plt.scatter(Train['spy'], Train['PredictedY'])
plt.xlabel("Actual SPY Change")
plt.ylabel("Predicted SPY Change")
plt.title("MLR Model Predictions vs Actual Values")
plt.show()

# Signal-Based Strategy Implementation

# Generate Trading Orders
Train['Order'] = [1 if sig > 0 else -1 for sig in Train['PredictedY']]
Train['Profit'] = Train['spy'] * Train['Order']
Train['Wealth'] = Train['Profit'].cumsum()
print('Total profit made in Train: ', Train['Profit'].sum())

# Plot Strategy Performance in Train Data
plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Train')
plt.plot(Train['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(Train['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()

# Apply Strategy to Test Data
Test['Order'] = [1 if sig > 0 else -1 for sig in Test['PredictedY']]
Test['Profit'] = Test['spy'] * Test['Order']
Test['Wealth'] = Test['Profit'].cumsum()
print('Total profit made in Test: ', Test['Profit'].sum())

# Plot Strategy Performance in Test Data
plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Test')
plt.plot(Test['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(Test['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()

# Model Evaluation using Sharpe Ratio and Maximum Drawdown
Train['Return'] = np.log(Train['Wealth']) - np.log(Train['Wealth'].shift(1))
Test['Return'] = np.log(Test['Wealth']) - np.log(Test['Wealth'].shift(1))

print('Train Daily Sharpe Ratio:', Train['Return'].mean() / Train['Return'].std(ddof=1))
print('Test Daily Sharpe Ratio:', Test['Return'].mean() / Test['Return'].std(ddof=1))

# Compute Maximum Drawdown
Train['Peak'] = Train['Wealth'].cummax()
Train['Drawdown'] = (Train['Peak'] - Train['Wealth']) / Train['Peak']
Test['Peak'] = Test['Wealth'].cummax()
Test['Drawdown'] = (Test['Peak'] - Test['Wealth']) / Test['Peak']

print('Maximum Drawdown in Train:', Train['Drawdown'].max())
print('Maximum Drawdown in Test:', Test['Drawdown'].max())
