import pandas as pd
import numpy as np

# ======== CONFIGURATION ========
CSV_FILES = ['EURUSD_YahooFinance_1h.csv', 'BTCUSDT_Binance_1h.csv', 'ETHUSDT_Binance_1h.csv', 'GBPUSD_YahooFinance_1h.csv', 'AUDUSD_YahooFinance_1h.csv', 'USDJPY_YahooFinance_1h.csv', 'USDCHF_YahooFinance_1h.csv', 'USDCAD_YahooFinance_1h.csv']  # Add more CSV files as needed
RISK_PERCENTAGE = 2  # Risk per trade (percentage of balance)
REWARD_RATIO = 6  # Reward-to-risk ratio
LIQUIDITY_WINDOW = 15  # Window size for calculating liquidity zones
SESSION_TIMEZONE = 'America/New_York'  # Timezone for session identification

# ======== FUNCTIONS ========

def load_data(file_name):
    """Load and preprocess data from a CSV file."""
    print(f"Loading data from {file_name}...")
    try:
        data = pd.read_csv(file_name)
        print(f"Data loaded successfully: {file_name}")
        
        # Identify and rename datetime column
        datetime_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if datetime_cols:
            data.rename(columns={datetime_cols[0]: 'Datetime'}, inplace=True)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
        else:
            raise ValueError("No datetime column found in the file.")

        # Flatten MultiIndex columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns]

        # Map required columns
        column_mappings = {'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}
        for key, value in column_mappings.items():
            matching_cols = [col for col in data.columns if key in col]
            if matching_cols:
                data[value] = data[matching_cols[0]]

        return data
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

def calculate_liquidity_zones(data, window):
    """Calculate liquidity zones."""
    data['Liquidity High'] = data['High'].rolling(window=window).max()
    data['Liquidity Low'] = data['Low'].rolling(window=window).min()
    return data

def calculate_fair_value_gaps(data):
    """Identify Fair Value Gaps (FVG)."""
    data['FVG'] = (data['High'].shift(1) < data['Low'].shift(-1)) | (data['Low'].shift(1) > data['High'].shift(-1))
    return data

def identify_sessions(data):
    """Identify trading sessions (Asian, London, New York)."""
    def session_label(hour):
        if 0 <= hour < 8:
            return 'Asian'
        elif 8 <= hour < 16:
            return 'London'
        else:
            return 'New York'

    data['Session'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert(SESSION_TIMEZONE).dt.hour.map(session_label)
    return data

def apply_ict_strategy(data):
    """Apply ICT strategy to generate signals."""
    data = calculate_liquidity_zones(data, LIQUIDITY_WINDOW)
    data = calculate_fair_value_gaps(data)
    data = identify_sessions(data)

    data['ICT Signal'] = (data['Low'] < data['Liquidity Low']) & (data['Close'] > data['Liquidity Low']) | data['FVG']
    return data

def backtest(data, initial_balance=100000):
    """Perform backtesting."""
    balance = initial_balance
    trades = []
    total_risk = 0
    total_reward = 0

    for i in range(len(data)):
        if data['ICT Signal'].iloc[i]:
            entry_price = data['Close'].iloc[i]
            stop_loss = data['Liquidity Low'].iloc[i]
            take_profit = entry_price + (entry_price - stop_loss) * REWARD_RATIO
            risk = balance * (RISK_PERCENTAGE / 100)
            position_size = risk / (entry_price - stop_loss)

            trade = {
                'Datetime': data['Datetime'].iloc[i],
                'Entry': entry_price,
                'Stop Loss': stop_loss,
                'Take Profit': take_profit,
                'Position Size': position_size,
                'Session': data['Session'].iloc[i],
                'Balance Before': balance
            }

            if i + 1 < len(data):
                if data['High'].iloc[i + 1] >= take_profit:
                    trade['Result'] = 'TP'
                    balance += risk * REWARD_RATIO
                    total_reward += risk * REWARD_RATIO
                elif data['Low'].iloc[i + 1] <= stop_loss:
                    trade['Result'] = 'SL'
                    balance -= risk
                    total_risk += risk
                else:
                    trade['Result'] = 'None'

            trade['Balance After'] = balance
            trades.append(trade)

    final_roi = ((balance - initial_balance) / initial_balance) * 100
    return trades, balance, final_roi, total_risk, total_reward

# ======== MAIN ========

if __name__ == "__main__":
    all_trades = []
    summary = []

    for file in CSV_FILES:
        data = load_data(file)
        if data is not None:
            data = apply_ict_strategy(data)
            trades, final_balance, roi, total_risk, total_reward = backtest(data)

            # Summarize results for the current file
            summary.append({
                'File': file,
                'Final Balance': final_balance,
                'ROI (%)': roi,
                'Total Trades': len(trades),
                'Total Risk': total_risk,
                'Total Reward': total_reward
            })

            all_trades.extend(trades)

    # Save detailed trade results
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv('ict_backtest_results.csv', index=False)
        print("Detailed trade results saved to ict_backtest_results.csv")

    # Save performance summary
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('ict_backtest_performance.csv', index=False)
        print("Performance summary saved to ict_backtest_performance.csv")