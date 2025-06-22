import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download historical data for multiple assets
tickers = ['AAPL', 'MSFT']
start_date = '2020-01-01'
end_date = '2021-01-01'

# Initialize empty DataFrame
data = pd.DataFrame()

for ticker in tickers:
    ticker_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    close_series = ticker_data['Close']
    close_series.name = ticker  # Set the column name for joining
    if data.empty:
        data = pd.DataFrame(close_series)
    else:
        data = data.join(close_series, how='outer')

# Technical indicators
window_short = 15
window_long = 50
window_bb = 20
bb_std = 2

for ticker in tickers:
    # Moving averages
    data[f'{ticker}_short_ma'] = data[ticker].rolling(window=window_short).mean()
    data[f'{ticker}_long_ma'] = data[ticker].rolling(window=window_long).mean()
    # Bollinger Bands
    data[f'{ticker}_ma_bb'] = data[ticker].rolling(window=window_bb).mean()
    data[f'{ticker}_upper_bb'] = data[f'{ticker}_ma_bb'] + bb_std * data[ticker].rolling(window=window_bb).std()
    data[f'{ticker}_lower_bb'] = data[f'{ticker}_ma_bb'] - bb_std * data[ticker].rolling(window=window_bb).std()

# Generate signals for each asset and strategy
for ticker in tickers:
    # Moving average crossover
    data[f'{ticker}_ma_signal'] = 0
    data.loc[data[f'{ticker}_short_ma'] > data[f'{ticker}_long_ma'], f'{ticker}_ma_signal'] = 1
    data.loc[data[f'{ticker}_short_ma'] <= data[f'{ticker}_long_ma'], f'{ticker}_ma_signal'] = 0
    # Bollinger Bands
    data[f'{ticker}_bb_signal'] = 0
    data.loc[data[ticker] < data[f'{ticker}_lower_bb'], f'{ticker}_bb_signal'] = 1
    data.loc[data[ticker] > data[f'{ticker}_upper_bb'], f'{ticker}_bb_signal'] = -1

# Portfolio simulation parameters
initial_capital = 100000.0
transaction_cost = 0.001  # 0.1%
slippage = 0.0005  # 0.05% per trade
position_size = 0.5  # 50% of capital per trade

# Portfolio simulation
portfolio_value = [initial_capital]
portfolio = {ticker: 0 for ticker in tickers}
cash = initial_capital

for i in range(1, len(data)):
    current_cash = cash
    for ticker in tickers:
        # Moving average strategy
        if data[f'{ticker}_ma_signal'].iloc[i] != data[f'{ticker}_ma_signal'].iloc[i-1]:
            if data[f'{ticker}_ma_signal'].iloc[i] == 1:  # Buy
                shares = (current_cash * position_size) // (data[ticker].iloc[i] * (1 + transaction_cost + slippage))
                cost = shares * data[ticker].iloc[i] * (1 + transaction_cost + slippage)
                cash -= cost
                portfolio[ticker] += shares
            else:  # Sell
                cash += portfolio[ticker] * data[ticker].iloc[i] * (1 - transaction_cost - slippage)
                portfolio[ticker] = 0
        # Bollinger Bands strategy
        if data[f'{ticker}_bb_signal'].iloc[i] != data[f'{ticker}_bb_signal'].iloc[i-1]:
            if data[f'{ticker}_bb_signal'].iloc[i] == 1:  # Buy
                shares = (current_cash * position_size) // (data[ticker].iloc[i] * (1 + transaction_cost + slippage))
                cost = shares * data[ticker].iloc[i] * (1 + transaction_cost + slippage)
                cash -= cost
                portfolio[ticker] += shares
            elif data[f'{ticker}_bb_signal'].iloc[i] == -1:  # Sell
                cash += portfolio[ticker] * data[ticker].iloc[i] * (1 - transaction_cost - slippage)
                portfolio[ticker] = 0
    # Calculate portfolio value at end of day
    portfolio_value.append(cash + sum(portfolio[ticker] * data[ticker].iloc[i] for ticker in tickers))

# Portfolio DataFrame
portfolio_df = pd.DataFrame({'portfolio_value': portfolio_value}, index=data.index)
portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1

# Buy and hold strategy for comparison
buy_hold_returns = (data[tickers].sum(axis=1) / data[tickers].sum(axis=1).iloc[0]) - 1

# Performance metrics
portfolio_df['returns'] = pd.to_numeric(portfolio_df['returns'], errors='coerce')
risk_free_rate = 0.01 / 252
excess_returns = portfolio_df['returns'] - risk_free_rate
sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
annualized_return = ((portfolio_df['portfolio_value'].iloc[-1] / initial_capital) ** (252/len(portfolio_df))) - 1
drawdown = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].cummax()) - 1
max_drawdown = drawdown.min()

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(portfolio_df.index, portfolio_df['cumulative_returns'], label='Strategy Cumulative Returns')
plt.plot(buy_hold_returns.index, buy_hold_returns, label='Buy & Hold Cumulative Returns')
plt.title('Multi-Asset Multi-Strategy Trading Simulation (2020)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.figure(figsize=(14, 4))
plt.plot(drawdown.index, drawdown)
plt.title('Portfolio Drawdown')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.grid(True)
plt.show()

print(f"Strategy Final Return: {portfolio_df['cumulative_returns'].iloc[-1]:.2%}")
print(f"Buy & Hold Final Return: {buy_hold_returns.iloc[-1]:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
