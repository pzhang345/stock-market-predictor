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
START = LAGS                 # first day with LAGS returns of history


def build_market(prices):
    """Equal-weighted market index from the given tickers, as a price-like series.

    Series are right-aligned (all downloads end on the same day); on each day the
    market return is the mean daily return of the tickers that have data that day.
    """
    num_days = max(len(stock_prices) for stock_prices in prices.values())
    # One row per ticker, one column per daily return; NaN where a ticker has no data yet
    daily_returns = np.full((len(prices), num_days - 1), np.nan)
    for ticker_idx, stock_prices in enumerate(prices.values()):
        daily_returns[ticker_idx, num_days - len(stock_prices):] = stock_prices[1:] / stock_prices[:-1] - 1
    market_returns = np.nanmean(daily_returns, axis=0)
    return np.concatenate([[1.0], np.cumprod(1 + market_returns)])


def _series_features(series, days):
    """Last LAGS daily returns ending at each index in `days` (uses only series[..day])."""
    returns = series[1:] / series[:-1] - 1
    return [returns[days - k] for k in range(1, LAGS + 1)]


def create_features(prices, market, horizon=1):
    """Build one row per (ticker, prediction day) to predict whether the stock goes up.

    Returns X, y:
    X is the concatenation of the stock's and the market's LAGS daily returns
    y is 1 if the stock's return over the next `horizon` days is positive, 0 otherwise
    """
    X = []
    y = []
    for _, stock_prices in prices.items():
        # Indices into stock_prices of each day we make a prediction on
        stock_days = np.arange(START, len(stock_prices) - horizon)
        if len(stock_days) == 0:
            continue

        market_days = stock_days + len(market) - len(stock_prices)

        cols = _series_features(stock_prices, stock_days) + _series_features(market, market_days)
        stock_fwd = stock_prices[stock_days + horizon] / stock_prices[stock_days] - 1

        X.append(np.column_stack(cols))
        y.append((stock_fwd > 0).astype(int))

    return np.vstack(X), np.hstack(y)

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    return accuracy

if __name__ == "__main__":
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
