import os
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

from data_loader import fetch_adjusted_close

def load_raw(filename):
    df = pd.read_csv(filename, sep=',', header=0, engine='python')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df

def check_missing_values(df):
    cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    na_counts = df[cols].isna().sum()
    issues = na_counts[na_counts > 0]
    if issues.empty:
        print('No NaNs found')
    else:
        print('NaNs found:')
        print(issues.to_string())
    return issues

def check_dups_and_order(df):
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        print(f"  WARNING - {dup_count} duplicate timestamp(s). Dropping duplicates, keeping first.")
        df = df[~df.index.duplicated(keep='first')]
    else:
        print('No duplicates found')

    if not df.index.is_monotonic_increasing:
        print("  WARNING - index was not sorted. Sorting now.")
        df = df.sort_index()
    else:
        print("OK - index already sorted.")
    return df

def check_missing_trading_days(df, symbol="UNKNOWN", exchange='NYSE'):
    cal = mcal.get_calendar(exchange)
    schedule = cal.valid_days(start_date=df.index.min(), end_date=df.index.max())
    expected = pd.DatetimeIndex(schedule).tz_localize(None)
    actual = pd.DatetimeIndex(df.index)
    missing = expected.difference(actual)

    if len(missing) == 0:
        print("  OK - no missing trading days (weekends/holidays already excluded by the exchange calendar).")
        return missing

    pos = {d: i for i, d in enumerate(expected)}
    missing_sorted = sorted(missing)
    runs =  [[missing_sorted[0]]]
    for d in missing_sorted[1:]:
        if pos[d] == pos[runs[-1][-1]] + 1:
            runs[-1].append(d)
        else:
            runs.append([d])

    print(f"  WARNING - {len(missing)} missing trading day(s) in {len(runs)} gap(s):")
    for run in runs:
        severity = "LIKELY REAL ISSUE - re-fetch this range" if len(run) >= 3 else "minor, worth spot-checking"
        span = f"{run[0].date()}" if len(run) == 1 else f"{run[0].date()} -> {run[-1].date()}"
        print(f"    {span}  ({len(run)} day(s))  [{severity}]")
    return missing


def check_price_sanity(df):
    checks = {
        'high < low': df['high'] < df['low'],
        'high < open': df['high'] < df['open'],
        'high < close': df['high'] < df['close'],
        'low > open': df['low'] > df['open'],
        'low > close': df['low'] > df['close'],
        'non-positive price': (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1),
        'negative volume': df['volume'] < 0,
        'zero volume': df['volume'] == 0,
    }

    any_issue = False
    for name, mask in checks.items():
        n = int(mask.sum())
        if n > 0:
            any_issue = True
            sample = df.index[mask][:5].strftime('%Y-%m-%d').tolist()
            print(f"  WARNING - {name}: {n} row(s), e.g. {sample}")

    if not any_issue:
        print("  OK - all OHLCV sanity checks passed.")
    return checks

def add_adjusted_close(df, symbol):
    """
        Refetches the same date range with auto_adjust=True to get a split/dividend
        -adjusted close, and attaches it as 'adj_close'. We keep the raw OHLC as-is
        (so open/high/low/close still reflect what actually traded that day) and use
        adj_close only for return/target calculations, so a split doesn't look like
        a huge price drop to the model.
    """
    adj = fetch_adjusted_close(symbol, start=df.index.min(), end=df.index.max() + pd.Timedelta(days=1))
    df = df.join(adj, how='left')

    missing_adj = df['adj_close'].isna().sum()
    if missing_adj > 0:
        raise ValueError(f"{missing_adj} row(s) missing adj_close. Check adjusted price data instead of replacing with raw close.")

    # flag days where the adjustment ratio jumps - i.e. a split/dividend event
    ratio = df['adj_close'] / df['close']
    ratio_change = ratio.pct_change().abs()
    split_days = df.index[ratio_change > 0.01]
    if len(split_days) > 0:
        print(f"  Detected {len(split_days)} likely split/dividend adjustment point(s): "f"{[d.date().isoformat() for d in split_days][:5]}")
    else:
        print("  No split/dividend adjustment points detected in this range.")
    return df

def add_target(df, price_col='adj_close'):
    """
        target = 1 if next trading day's price is higher than today's, else 0.
        Uses shift(-1) so target[t] is computed from price[t+1] vs price[t] - this
        is intentionally the *only* place future data is allowed to touch a row.
        The final row has no "next day" yet, so its target is left as NaN - drop
        it before training rather than guessing.
    """
    df = df.copy()
    next_price = df[price_col].shift(-1)
    df['target'] = (next_price > df[price_col]).astype(int)
    df.loc[df.index[-1], 'target'] = np.nan
    return df


def data_cleanup(filename, symbol=None, exchange='NYSE', fetch_adjusted=True, output_dir="data/cleaned"):
    df = load_raw(filename)
    if symbol is None:
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "UNKNOWN"

    print(f"=== Cleaning {symbol} ({len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}) ===")

    print("\n[1/6] NaN check")
    missing_values = check_missing_values(df)
    if not missing_values.empty:
        raise ValueError("Missing OHLCV values found. Re-fetch or repair the source before training.")

    print("\n[2/6] Duplicate / order check")
    df = check_dups_and_order(df)

    print("\n[3/6] Missing trading days")
    missing_days = check_missing_trading_days(df, symbol, exchange)
    if len(missing_days) > 0:
        raise ValueError("Missing exchange trading days found. Re-fetch or verify those dates first.")

    print("\n[4/6] OHLCV sanity checks")
    price_checks = check_price_sanity(df)
    if any(mask.any() for mask in price_checks.values()):
        raise ValueError("Invalid OHLCV values found. Review the warnings and repair the raw data.")

    print("\n[5/6] Adjusted close (split/dividend handling)")
    if fetch_adjusted:
        # An error here is deliberate: raw close may distort returns around splits.
        df = add_adjusted_close(df, symbol)
    else:
        print("  WARNING - using unadjusted close explicitly (fetch_adjusted=False).")
        df['adj_close'] = df['close']

    print("\n[6/6] Target column")
    price_col = 'adj_close'
    df = add_target(df, price_col=price_col)
    print(f"  Target built from '{price_col}'. Class balance: "
          f"{df['target'].value_counts(dropna=True).to_dict()}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{symbol}_cleaned.csv")
    df.to_csv(out_path)
    print(f"\nSaved -> {out_path}")
    return df


if __name__ == "__main__":
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else "AAPL.csv"
    sym = sys.argv[2] if len(sys.argv) > 2 else None
    data_cleanup(fname, symbol=sym)



