from data_loader import get_data

# Download 2018-2023 inclusive; end is exclusive in yfinance.
# Set refresh=True when you need to update an already cached interval.
full_df = get_data("AAPL", start="2018-01-01", end="2024-01-01")
print(f"Downloaded/loaded {len(full_df)} rows ({full_df.index.min().date()} -> {full_df.index.max().date()})")
# Training / validation / testing is handled by train_models.py.
