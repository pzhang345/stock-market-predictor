""" Predict whether a stock beats the market over the next N days, using only data up to today.
The model is a Autoregressive model trained on the stock's own recent returns, momentum, price vs moving average, and volatility,
To run use:
python timeseries/model.py
"""

import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loadDataset import load_all

class TimeSeriesModel:
    def __init__(self):
        self.weights = None
    
    def _add_bias(self, X):
        """Prepend a column of ones so weights[0] acts as the bias/intercept."""
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xb = self._add_bias(X)
        self.weights = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y

    def score(self, X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Please call the 'fit' method before making predictions.")
        return self._add_bias(X) @ self.weights

    def predict_ranked(self, X, days):
        """On each day, predict 1 ("beats the market") for the stocks whose score is
        above that day's median score, 0 for the rest. Ties at the median go to 0.

        `days` gives the calendar day of each row (same length as X), so stocks are
        only compared with other stocks on the same day.
        """
        scores = self.score(X)
        preds = np.zeros(len(scores), dtype=int)
        for d in np.unique(days):
            idx = np.flatnonzero(days == d)
            preds[idx] = scores[idx] > np.median(scores[idx])
        return preds

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Please call the 'fit' method before making predictions.")
        return (self._add_bias(X) @ self.weights > 0.5).astype(int)

LAGS = 5                    # recent daily returns
MOMENTUM = (20, 60, 252)    # total return over the past ~1, 3, 12 months
MA_WINDOWS = (20, 50)       # price relative to its moving average
VOL_WINDOW = 20             # std of daily returns over the past ~month
START = max(LAGS, *MOMENTUM, *MA_WINDOWS, VOL_WINDOW)  # first day with full history


def build_market(prices):
    """Equal-weighted market index from the given tickers, as a price-like series.

    Series are right-aligned (all downloads end on the same day); on each day the
    market return is the mean daily return of the tickers that have data that day.
    """
    M = max(len(p) for p in prices.values())
    rets = np.full((len(prices), M - 1), np.nan)
    for i, p in enumerate(prices.values()):
        rets[i, M - len(p):] = p[1:] / p[:-1] - 1
    market_r = np.nanmean(rets, axis=0)
    return np.concatenate([[1.0], np.cumprod(1 + market_r)])


def _series_features(p, e):
    """Feature columns for price series p at end-of-window indices e (uses only p[..e])."""
    r = p[1:] / p[:-1] - 1
    cols = [r[e - k] for k in range(1, LAGS + 1)]
    cols += [p[e] / p[e - L] - 1 for L in MOMENTUM]
    cs = np.concatenate([[0.0], np.cumsum(p)])
    cols += [p[e] / ((cs[e + 1] - cs[e + 1 - W]) / W) - 1 for W in MA_WINDOWS]
    cols.append(sliding_window_view(r, VOL_WINDOW).std(axis=1)[e - VOL_WINDOW])
    return cols


def create_features(prices, market, horizon=1):
    """Build one row per (ticker, day e) to predict whether the stock beats the market.

    Features (only data up to and including day e):
      - the stock's own: last LAGS daily returns, momentum p[e]/p[e-L]-1,
        price vs moving average p[e]/MA_W-1, VOL_WINDOW-day volatility
      - the same features computed on the market index (from build_market)
    Label: 1 if the stock's return from day e to e+horizon is greater than the
    market's return over the same days.

    Rows start at e=START so every feature has its full history.
    Returns X, y, days where days[i] is row i's day as an index into `market`,
    so rows from different stocks on the same calendar day share a value.
    """
    X = []
    y = []
    days = []
    for ticker, p in prices.items():
        e = np.arange(START, len(p) - horizon)
        if len(e) == 0:
            continue
        em = e + len(market) - len(p) 

        cols = _series_features(p, e) + _series_features(market, em)
        stock_fwd = p[e + horizon] / p[e] - 1
        market_fwd = market[em + horizon] / market[em] - 1

        X.append(np.column_stack(cols))
        y.append((stock_fwd > market_fwd).astype(int))
        days.append(em)

    return np.vstack(X), np.hstack(y), np.hstack(days)

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    return accuracy

def time_split(prices, market, frac=0.7):
    """Split every series at the same calendar day: first `frac` of days for training,
    the rest for testing. The test slice keeps START days of history before the cut so
    its first row is the first day after the cut; training labels all end before the cut."""
    M = len(market)
    drop = M - int(M * frac)
    head = {k: p[:len(p) - drop] for k, p in prices.items() if len(p) - drop > START}
    tail = {k: p[len(p) - drop - START:] for k, p in prices.items() if len(p) - drop - START >= 0}
    return head, tail, market[:M - drop]


if __name__ == "__main__":
    prices = load_all()
    market = build_market(prices)
    train, test, train_market = time_split(prices, market)

    print(f"{'horizon':>8} {'accuracy':>11}")
    for horizon in [1, 5, 10, 20, 60, 252]:
        train_X, train_y, _ = create_features(train, train_market, horizon=horizon)
        test_X, test_y, test_days = create_features(test, market, horizon=horizon)

        model = TimeSeriesModel()
        model.fit(train_X, train_y)
        eval = evaluate_model(model, test_X, test_y)
        print(f"{horizon:>8} {eval:>11.4f} {'':>7} {'':>11} {'':>9} {'':>16}")
        base = np.mean(test_y == int(train_y.mean() > 0.5))
