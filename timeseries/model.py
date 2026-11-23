""" Predict whether a stock beats is positive over the next N days, using only data up to today.
The model is an autoregressive model trained on the stock's and the market's daily returns.
To run use:
python timeseries/model.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loadDataset import load_train, load_test

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

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Please call the 'fit' method before making predictions.")
        return (self._add_bias(X) @ self.weights > 0.5).astype(int)

LAGS = 100                   # number of past daily % changes to use
START = LAGS                # first day with LAGS returns of history


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
    """Last LAGS daily returns ending at day e (uses only p[..e])."""
    r = p[1:] / p[:-1] - 1
    return [r[e - k] for k in range(1, LAGS + 1)]


def create_features(prices, market, horizon=1):
    """Build one row per (ticker, day e) to predict whether the stock beats the market.

    Returns X, y, days, where `days` is the calendar day of each row (same length as X).
    X is the concatenation of the stock's and the market's LAGS daily returns, 
    y is 1 if the stock beats the market over the next `horizon` days, 0 otherwise.
    """
    X = []
    y = []
    for ticker, p in prices.items():
        e = np.arange(START, len(p) - horizon)
        if len(e) == 0:
            continue
        em = e + len(market) - len(p) 

        cols = _series_features(p, e) + _series_features(market, em)
        stock_fwd = p[e + horizon] / p[e] - 1

        X.append(np.column_stack(cols))
        y.append((stock_fwd > 0).astype(int))

    return np.vstack(X), np.hstack(y)

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    return accuracy

if __name__ == "__main__":
    # Train on trainTickers.txt, test on testTickers.txt (held-out stocks).
    train = load_train()
    test = load_test()
    train_market = build_market(train)
    market = build_market({**train, **test})

    print(f"{'horizon':>8} {'accuracy':>11}")
    for horizon in [1, 5, 10, 20, 60, 252]:
        train_X, train_y = create_features(train, train_market, horizon=horizon)
        test_X, test_y = create_features(test, market, horizon=horizon)

        model = TimeSeriesModel()
        model.fit(train_X, train_y)
        acc = evaluate_model(model, test_X, test_y)
        print(f"{horizon:>8} {acc:>11.4f}")
