import pandas as pd
import requests
import yfinance as yf
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

# ======== CONFIGURATION ========
FOREX_PAIRS = ['USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'XAUUSD=X']  # Add multiple Yahoo Finance symbols
CRYPTO_SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # Add multiple Binance symbols

# Date Ranges
FOREX_START_DATE = '2023-01-01'
FOREX_END_DATE = '2024-12-01'
CRYPTO_START_DATE = '2020-01-01'
CRYPTO_END_DATE = '2024-12-01'

# Intervals
FOREX_INTERVAL = '1h'  # Supported intervals: '1h', '4h', '1d'
CRYPTO_INTERVAL = '1h'  # Supported intervals: '1h', '4h'
MAX_RETRIES = 5  # Maximum retry attempts for Binance

# Toggle crypto fetching
ENABLE_CRYPTO_FETCH = False  # Set to True to enable crypto fetching

# ======== FUNCTIONS ========

def save_data_to_csv(data, symbol, source):
    """Save data to a CSV file named after the symbol and source."""
    if data is not None and not data.empty:
        filename = f"{symbol}_{source}_{CRYPTO_INTERVAL}.csv"
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print(f"No data to save for {symbol} from {source}.")

def fetch_forex_data_yahoo(pair, start_date, end_date, interval):
    """Fetch Forex data from Yahoo Finance."""
    print(f"Fetching Forex data for {pair} from Yahoo Finance...")
    try:
        data = yf.download(pair, start=start_date, end=end_date, interval=interval)
        if not data.empty:
            print(f"Data fetched successfully for {pair}.")
            data.reset_index(inplace=True)
            return data
        else:
            print(f"No data found for {pair}.")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def fetch_crypto_data_binance(symbol, interval, start_date, end_date, retries=MAX_RETRIES):
    """Fetch cryptocurrency data from Binance API with pagination."""
    print(f"Fetching crypto data for {symbol} from Binance...")
    all_data = []
    start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)

    while start_timestamp < end_timestamp:
        attempt = 0
        while attempt < retries:
            try:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_timestamp,
                    "endTime": end_timestamp,
                    "limit": 1000,
                }
                response = requests.get(url, params=params)
                data = response.json()

                if isinstance(data, list):
                    df = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume",
                                                     "Close time", "Quote asset volume", "Number of trades",
                                                     "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])
                    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
                    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
                    df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
                    all_data.append(df)

                    # Update the start timestamp for the next request
                    start_timestamp = int(df["Open time"].iloc[-1].timestamp() * 1000) + 1
                    break
                else:
                    print(f"Binance error: {data.get('msg', 'No data available')}")
                    attempt += 1
            except Exception as e:
                print(f"Error fetching data: {e}")
                attempt += 1

        if attempt >= retries:
            print(f"Max retries reached for Binance data for {symbol}. Stopping further requests.")
            break

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        result_df = result_df[(result_df["Open time"] >= pd.to_datetime(start_date)) &
                              (result_df["Open time"] <= pd.to_datetime(end_date))]
        return result_df
    else:
        print(f"No data fetched for {symbol}.")
        return None

def add_technical_indicators(data, symbol):
    """Add VWAP, RSI, and Bollinger Bands to the dataset."""
    print("Adding technical indicators...")

    # Flatten MultiIndex columns if applicable
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(filter(None, col)).strip() for col in data.columns]
        print("Flattened Columns:", data.columns)

    # Dynamically generate column names for the current symbol
    base_columns = ['High', 'Low', 'Close', 'Volume']
    if "Open time" in data.columns:  # Adjust for crypto data
        required_columns = base_columns
    else:  # Adjust for forex data
        required_columns = [f"{col}_{symbol}=X" for col in base_columns]

    # Ensure required columns exist and are numeric
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")
        
        # Convert to numeric explicitly and handle errors
        data[column] = pd.to_numeric(data[column], errors='coerce')
        if data[column].isnull().all():
            print(f"WARNING: Column {column} contains all NaN values after conversion.")
            raise ValueError(f"Column {column} contains invalid data.")

    # Calculate VWAP
    if "Open time" in data.columns:  # Crypto VWAP
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        data['VWAP'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    else:  # Forex VWAP
        data['Typical Price'] = (data[f'High_{symbol}=X'] + data[f'Low_{symbol}=X'] + data[f'Close_{symbol}=X']) / 3
        data['Cumulative Volume'] = data[f'Volume_{symbol}=X'].cumsum()
        data['Cumulative TPV'] = (data['Typical Price'] * data[f'Volume_{symbol}=X']).cumsum()
        data['VWAP'] = data['Cumulative TPV'] / data['Cumulative Volume']

    # Drop intermediate columns
    data.drop(columns=['Typical Price', 'Cumulative Volume', 'Cumulative TPV'], inplace=True, errors='ignore')

    # Calculate RSI
    print("Calculating RSI...")
    rsi_col = 'Close' if "Open time" in data.columns else f'Close_{symbol}=X'
    rsi = RSIIndicator(close=data[rsi_col], window=14)
    data['RSI'] = rsi.rsi()

    # Calculate Bollinger Bands
    print("Calculating Bollinger Bands...")
    bb = BollingerBands(close=data[rsi_col], window=20, window_dev=2)
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()

    print("Technical indicators added successfully.")
    return data

# ======== MAIN ========

if __name__ == "__main__":
    # Fetch Forex data
    for pair in FOREX_PAIRS:
        symbol = pair.replace('=X', '')  # Extract the base symbol name (e.g., EURUSD)
        forex_data = fetch_forex_data_yahoo(pair, FOREX_START_DATE, FOREX_END_DATE, FOREX_INTERVAL)
        if forex_data is not None:
            forex_data = add_technical_indicators(forex_data, symbol)
            save_data_to_csv(forex_data, symbol, "YahooFinance")
            print(f"Processed Forex data for {pair}.")

    # Fetch Crypto data (conditionally)
    if ENABLE_CRYPTO_FETCH:
        for symbol in CRYPTO_SYMBOLS:
            crypto_data = fetch_crypto_data_binance(symbol, CRYPTO_INTERVAL, CRYPTO_START_DATE, CRYPTO_END_DATE)
            if crypto_data is not None:
                crypto_data = add_technical_indicators(crypto_data, symbol)
                save_data_to_csv(crypto_data, symbol, "Binance")
                print(f"Processed Crypto data for {symbol}.")