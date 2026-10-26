import yfinance as yf
import pandas as pd
import os

def daily_fetch_data(symbol, start=None, end=None):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for symbol='{symbol}'. Check the ticker is correct.")

    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
    })
    df.index.name = 'timestamp'
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]]
    df["symbol"] = symbol
    return df


def fetch_adjusted_close(symbol, start=None, end=None):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No adjusted-close data returned for symbol='{symbol}'.")

    df = df.rename(columns={'Close': 'adj_close'})
    df.index.name = 'timestamp'
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[['adj_close']]


def get_data(symbol, start=None, end=None, cache_dir="data/raw", refresh=False):
    """Load a date-specific cache, or download the requested time interval.

    yfinance treats `end` as exclusive. Pass refresh=True to force a re-download.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_key = pd.Timestamp(start).strftime("%Y%m%d") if start is not None else "all"
    end_key = pd.Timestamp(end).strftime("%Y%m%d") if end is not None else "latest"
    path = os.path.join(cache_dir, f"{symbol}_{start_key}_{end_key}.csv")

    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "timestamp"
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        return df

    df = daily_fetch_data(symbol, start=start, end=end)
    df.to_csv(path, index=True)
    return df


def get_multiple(symbols, start=None, end=None, cache_dir="data/raw", refresh=False):
    frames = []
    for sym in symbols:
        frames.append(get_data(sym, start=start, end=end,
                               cache_dir=cache_dir, refresh=refresh))
    return pd.concat(frames)
