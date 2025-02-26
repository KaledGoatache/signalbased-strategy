# Signal-Based Trading Strategy with MLR

### Description
This project applies **Multiple Linear Regression (MLR)** to predict and trade the **SPY ETF**, leveraging global stock indices. It evaluates the **Signal-Based Trading Strategy** against the **Buy & Hold** approach using key financial metrics.

### Key Features
- **Data Processing**: Extracts open/close prices from global indices.
- **Feature Engineering**: Includes lagged variables and inter-market relationships.
- **MLR Model**: Estimates SPY price movement using **Ordinary Least Squares (OLS)**.
- **Trading Strategy**: Generates buy/sell signals based on predictions.
- **Performance Metrics**:
  - **Sharpe Ratio** (Risk-Adjusted Returns)
  - **Maximum Drawdown** (Risk Exposure)

### Libraries Used
- `pandas`, `numpy` - Data manipulation
- `statsmodels` - Regression modeling
- `matplotlib` - Visualization

## MLR Model & Insights
- The **Multiple Linear Regression model** aimed to find relationships between SPY price movement and various global indices.
- **Results:**
  - **Train R²: 0.059 | Test R²: 0.067** → The model has **low explanatory power**, meaning it doesn’t strongly predict SPY price changes.
  - **Train RMSE: 1.22 | Test RMSE: 1.70** → The model has notable prediction errors but still provides directional insight.
  - **Some features, such as AORD (Australia) and DAXI (Germany), showed statistical significance**, suggesting they may impact SPY’s movement.
  - Despite the weak predictive power, the model was still used to generate **trading signals**, showing that a **slight statistical edge can still be profitable** over many trades.

## Trading Strategy
- The model predictions were used to develop a **Signal-Based Trading Strategy**, which executes trades based on the sign of `PredictedY`.
- **Buy (`Order = 1`)** if predicted return is **positive**.
- **Sell (`Order = -1`)** if predicted return is **negative**.

Profit Calculation:
```math
Profit = spy * Order
```
Cumulative Wealth Calculation:
```math
Wealth = \sum Profit
```

### Performance Metrics
- **Sharpe Ratio:**
  ```math
  Sharpe Ratio = Mean(Return) / StdDev(Return)
  ```
- **Maximum Drawdown:**
  ```math
  Drawdown = (Peak - Wealth) / Peak
  ```

## Results
- **Train Profit: 214.34 | Test Profit: 241.03**
- **Sharpe Ratio (Train: 2.85 | Test: 2.07)** → Strong risk-adjusted returns.
- **Maximum Drawdown (Train: 6.06% | Test: 11.72%)** → Higher risk in test data.
- **Signal-Based Strategy outperforms Buy & Hold** in both datasets.

## Summary
This project demonstrates how **MLR can be used for financial forecasting and trading strategies**. While the **MLR model itself had weak predictive power**, it was still able to **generate profitable trading signals**. The **Signal-Based Strategy** showed **higher profitability and better risk-adjusted returns** than Buy & Hold.

