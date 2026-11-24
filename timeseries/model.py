""" Predict whether a stock's return over the next day is positive, using only data up to today.
The model is an autoregressive model trained on the stock's and the market's daily returns.

To run use:
python timeseries/model.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

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

    train_X, train_y = create_features(train, train_market)
    test_X, test_y = create_features(test, market)

    print(f"Train: {train_X.shape[0]} samples, {train_X.shape[1]} features")
    print(f"Test: {test_X.shape[0]} samples, {test_X.shape[1]} features")
    model = TimeSeriesModel()
    model.fit(train_X, train_y)
    acc = evaluate_model(model, test_X, test_y)
    print(f"Test accuracy: {acc:.4f}")

    # Rows are true labels, columns are predictions (0 = down, 1 = up)
    mtrx = confusion_matrix(test_y, model.predict(test_X))
    print("Confusion matrix:")
    print(mtrx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(test_y, model.score(test_X), name="Linear AR", plot_chance_level=True, ax=ax1)
    ax1.set_title("ROC curve – next-day direction")

    ConfusionMatrixDisplay(confusion_matrix=mtrx, display_labels=["Down", "Up"]).plot(ax=ax2, cmap="Blues", colorbar=False)
    ax2.set_title("Confusion matrix")

    plt.tight_layout()
    plt.show()
