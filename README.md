# Multi-Asset Algorithmic Trading Simulation

## Overview

This project simulates algorithmic trading strategies (moving average crossover and Bollinger Bands) across multiple equities using historical market data. It incorporates realistic portfolio management features, including transaction costs, position sizing, and advanced performance analytics.

## Features

- **Multi-asset trading:** Supports trading on multiple equities (e.g., AAPL, MSFT).
- **Strategy backtesting:** Implements and evaluates moving average crossover and Bollinger Bands strategies.
- **Portfolio management:** Includes transaction costs, slippage, and position sizing.
- **Performance analytics:** Computes Sharpe ratio, annualized return, and maximum drawdown.
- **Visualization:** Plots cumulative returns and portfolio drawdown for strategy evaluation.

## Tech Stack

- **Python**
- **pandas**
- **yfinance**
- **NumPy**
- **Matplotlib**

## Usage

1. **Install dependencies:**
`pip install yfinance pandas numpy matplotlib`
2. **Run the simulation:**
`python main.py`
3. **View results:**  
The script outputs performance metrics and plots equity curves and drawdown.
