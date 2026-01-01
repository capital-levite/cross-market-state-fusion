import numpy as np
import pandas as pd
import os
import sys

def calculate_metrics(df):
    if df.empty:
        return 0, 0, 0
    
    # PnL is usually in a 'pnl' column
    if 'pnl' not in df.columns:
        print(f"Warning: 'pnl' column not found. Columns: {df.columns}")
        return 0, 0, 0
    
    pnl = df['pnl'].values
    cumulative_pnl = np.cumsum(pnl)
    
    # Sharpe Ratio (assuming 0 risk-free rate)
    # Annualized for 15m intervals (approx 35040 intervals per year)
    returns = pnl
    if len(returns) < 2 or np.std(returns) == 0:
        sharpe = 0
    else:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(35040)
    
    # Ulcer Index
    peak = np.maximum.accumulate(cumulative_pnl)
    # Drawdown is cumulative_pnl - peak (negative or zero)
    drawdown = cumulative_pnl - peak
    # Ulcer Index is sqrt(mean(drawdown^2))
    ulcer_index = np.sqrt(np.mean(drawdown**2))
    
    return sharpe, ulcer_index, len(df)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <csv_file>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    try:
        df = pd.read_csv(csv_file)
        sharpe, ulcer, trades = calculate_metrics(df)
        
        print(f"File: {os.path.basename(csv_file)}")
        print(f"Trades: {trades}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Ulcer Index: {ulcer:.4f}")
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
