import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

def load_stock_data(file_path):
    """Load stock data from CSV file"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def calculate_technical_indicators(df):
    """Calculate various technical indicators using TA-Lib"""
    # Moving Averages
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)

    # RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # MACD (Moving Average Convergence Divergence)
    macd, macd_signal, macd_hist = talib.MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # Volume indicators
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    return df

def plot_technical_analysis(df, symbol):
    """Create visualizations for technical analysis"""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # Price and Moving Averages
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.plot(df.index, df['SMA_20'], label='20-day SMA')
    ax1.plot(df.index, df['SMA_50'], label='50-day SMA')
    ax1.plot(df.index, df['SMA_200'], label='200-day SMA')
    ax1.set_title(f'{symbol} Price and Moving Averages')
    ax1.legend()
    ax1.grid(True)

    # RSI
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(df.index, df['RSI'], label='RSI')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend()
    ax2.grid(True)

    # MACD
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(df.index, df['MACD'], label='MACD')
    ax3.plot(df.index, df['MACD_Signal'], label='Signal Line')
    ax3.bar(df.index, df['MACD_Hist'], label='MACD Histogram')
    ax3.set_title('MACD')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f'analysis_results/{symbol}_technical_analysis.png')
    plt.close()

def calculate_financial_metrics(df):
    """Calculate key financial metrics"""
    # Daily returns
    df['Returns'] = df['Close'].pct_change()

    # Volatility (20-day rolling standard deviation)
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)

    # Sharpe Ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    excess_returns = df['Returns'] - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    # Maximum Drawdown
    rolling_max = df['Close'].expanding().max()
    drawdowns = df['Close']/rolling_max - 1.0
    max_drawdown = drawdowns.min()

    return {
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Volatility': df['Volatility'].iloc[-1],
        'Average_Daily_Return': df['Returns'].mean(),
        'Total_Return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    }

def main():
    # Create directory for analysis results
    Path('analysis_results').mkdir(exist_ok=True)

    # List of stock symbols to analyze
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA']

    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")

        # Load data
        file_path = f'../yfinance_data/{symbol}_historical_data.csv'
        df = load_stock_data(file_path)

        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Plot technical analysis
        plot_technical_analysis(df, symbol)

        # Calculate and print financial metrics
        metrics = calculate_financial_metrics(df)
        print(f"\nFinancial Metrics for {symbol}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
