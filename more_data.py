import os
import pandas as pd
import numpy as np

def load_cleaned_file(filename):
    df = pd.read_csv(filename, sep=',', header=0, engine='python')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df

def add_returns(df):
    """
    Adds price percentage difference from previous day and also lag from last three days.
    """
    df = df.copy()
    df['return_1d'] = df['adj_close'].pct_change()
    df['return_1d_1lag'] = df['return_1d'].shift(1)
    df['return_1d_2lag'] = df['return_1d'].shift(2)
    df['return_1d_3lag'] = df['return_1d'].shift(3)

    return df

def add_moving_average(df):
    """
    Calculates average price from set number of days.
    And then adds columns for how much is price up or down compared to the average.
    """
    df = df.copy()
    sma_5 = df['adj_close'].rolling(window=5).mean()
    sma_10 = df['adj_close'].rolling(window=10).mean()
    sma_20 = df['adj_close'].rolling(window=20).mean()

    df['price_to_SMA_5'] = df['adj_close'] / sma_5
    df['price_to_SMA_10'] = df['adj_close'] / sma_10
    df['price_to_SMA_20'] = df['adj_close'] / sma_20
    return df

def add_relative_str_index(df):
    df = df.copy()
    delta = df['adj_close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df

def exponential_moving_average(df):
    df = df.copy()
    ema_12 = df['adj_close'].ewm(span=12).mean()
    ema_26 = df['adj_close'].ewm(span=26).mean()

    df['macd'] = ema_12 - ema_26

    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    return df

def volatility(df):
    df = df.copy()
    df['volatility_10'] = df['return_1d'].rolling(window=10).std()
    return df

def volume_change(df):
    df = df.copy()
    df['volume_change'] = df['volume'].pct_change()
    return df


def build_features(filename, symbol, output_dir="data/features"):
    df = load_cleaned_file(filename)

    df = add_returns(df)
    df = add_moving_average(df)
    df = add_relative_str_index(df)
    df = exponential_moving_average(df)
    df = volatility(df)
    df = volume_change(df)

    # Preserve rows whose future target is not known yet; do not discard
    # them merely because the last observation has target=NaN.
    feature_cols = [
        'return_1d', 'return_1d_1lag', 'return_1d_2lag', 'return_1d_3lag',
        'price_to_SMA_5', 'price_to_SMA_10', 'price_to_SMA_20',
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'volatility_10', 'volume_change',
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{symbol}_features.csv")
    df.to_csv(out_path)
    print(f"Saved -> {out_path} ({len(df)} rows after dropping NaN warmup rows)")
    return df

if __name__ == "__main__":
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else "data/cleaned/AAPL_cleaned.csv"
    sym = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
    df = build_features(fname, sym)
    print(df.tail())
