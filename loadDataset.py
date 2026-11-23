"""Load daily closing prices for the train/test ticker lists.

Usage:
    from loadDataset import load_train, load_test
    train = load_train()          # {"AAPL": np.array([...]), ...}
    test = load_test()
"""

import os
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "data")


def read_tickers(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _download(tickers, start, end):
    """Return {ticker: np.ndarray of closes}; missing tickers are omitted."""
    out = {}
    df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    for t in tickers:
        if t in df:
            s = df[t].dropna()
            if len(s):
                out[t] = s.to_numpy(dtype=np.float64)
    return out


def load_prices(ticker_file, years=10, refresh=False, retries=2):
    name = os.path.splitext(os.path.basename(ticker_file))[0]
    cache = os.path.join(CACHE_DIR, f"{name}_{years}y.npz")
    if os.path.exists(cache) and not refresh:
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}

    tickers = read_tickers(ticker_file)
    end = datetime.today()
    start = end - timedelta(days=round(365.25 * years))
    prices = _download(tickers, start, end)

    # Batch downloads sometimes drop tickers (rate limits / sqlite locks); retry those.
    for _ in range(retries):
        missing = [t for t in tickers if t not in prices]
        if not missing:
            break
        prices.update(_download(missing, start, end))

    missing = [t for t in tickers if t not in prices]
    if missing:
        print(f"[loadDataset] no data for {len(missing)} tickers: {missing}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, **prices)
    return prices


def load_train(**kwargs):
    return load_prices(os.path.join(ROOT, "trainTickers.txt"), **kwargs)


def load_test(**kwargs):
    return load_prices(os.path.join(ROOT, "testTickers.txt"), **kwargs)

def load_all(**kwargs):
    """Load both train and test datasets."""
    return {**load_train(**kwargs), **load_test(**kwargs)}