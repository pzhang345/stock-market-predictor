"""
Bayesian logistic regression: will a stock have a positive gain over the next N days?
Likelihood: y_i ~ Bern(sigmoid(w^T x_i)).  Prior: w ~ N(0, PRIOR_SD^2 I).
The posterior has no closed form, so we sample it with Metropolis (isotropic Gaussian proposal).

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loadDataset import load_train, load_test

N = 20
STEP = 20
PRIOR_SD = 10
PROPOSAL_SD = 0.008
NUM_SAMPLES = 10000
BURN_IN = int(0.18 * NUM_SAMPLES)
FEATURES = ["intercept", "ret_20d", "ret_252d", "vol_252d", "mkt_ret_20d"]


class BayesianLogisticRegression:
    def __init__(self, prior_sd=PRIOR_SD, proposal_sd=PROPOSAL_SD,
                 num_samples=NUM_SAMPLES, burn_in=BURN_IN, seed=0):
        self.prior_sd = prior_sd
        self.proposal_sd = proposal_sd
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.samples = None

    def log_posterior(self, w, X, y):
        """log p(w|X,y) up to a constant: Bernoulli log-likelihood + Normal log-prior."""
        z = X @ w
        log_lik = np.sum(y * z - np.logaddexp(0, z))
        log_prior = -np.sum(w ** 2) / (2 * self.prior_sd ** 2)
        return log_lik + log_prior

    def fit(self, X, y):
        """Sample the posterior with Metropolis (isotropic Gaussian proposal)."""
        w = np.zeros(X.shape[1])
        current = self.log_posterior(w, X, y)
        self.all_samples, accepted = np.zeros((self.num_samples, len(w))), 0
        for i in range(self.num_samples):
            w_prop = w + self.rng.normal(0, self.proposal_sd, size=len(w))
            proposed = self.log_posterior(w_prop, X, y)
            if np.log(self.rng.uniform()) < proposed - current:
                w, current = w_prop, proposed
                accepted += 1
            self.all_samples[i] = w
        self.acceptance_rate = accepted / self.num_samples
        self.samples = self.all_samples[self.burn_in:]

    def predict_proba(self, X):
        """Posterior predictive P(up): sigmoid(w^T x) averaged over posterior samples."""
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Please call 'fit' first.")
        return np.mean(1 / (1 + np.exp(-X @ self.samples.T)), axis=1)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


def build_market(prices):
    """Equal-weighted market index (same as timeseries/model.py)."""
    num_days = max(len(p) for p in prices.values())
    daily_returns = np.full((len(prices), num_days - 1), np.nan)
    for i, p in enumerate(prices.values()):
        daily_returns[i, num_days - len(p):] = p[1:] / p[:-1] - 1
    return np.concatenate([[1.0], np.cumprod(1 + np.nanmean(daily_returns, axis=0))])


def create_features(prices, market):
    """One row every STEP days per stock, using only prices up to that day.
    y = 1 if the stock is higher N days later, 0 otherwise."""
    X, y = [], []
    for p in prices.values():
        offset = len(market) - len(p)
        for t in range(252, len(p) - N, STEP):
            m = t + offset
            daily = p[t - 251:t + 1] / p[t - 252:t] - 1
            X.append([p[t] / p[t - 20] - 1,
                      p[t] / p[t - 252] - 1,
                      daily.std() * np.sqrt(252),
                      market[m] / market[m - 20] - 1])
            y.append(int(p[t + N] > p[t]))
    return np.array(X), np.array(y)


if __name__ == "__main__":
    train = load_train()
    test = load_test()
    X_train, y_train = create_features(train, build_market(train))
    X_test, y_test = create_features(test, build_market({**train, **test}))

    # Standardize features with training stats, then add the intercept column
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = np.column_stack([np.ones(len(X_train)), (X_train - mu) / sd])
    X_test = np.column_stack([np.ones(len(X_test)), (X_test - mu) / sd])

    model = BayesianLogisticRegression()
    model.fit(X_train, y_train)
    print(f"acceptance rate: {model.acceptance_rate:.3f}")

    print(f"\n{'feature':>12} {'mean':>8} {'2.5%':>8} {'97.5%':>8}")
    for j, name in enumerate(FEATURES):
        lo, hi = np.percentile(model.samples[:, j], [2.5, 97.5])
        print(f"{name:>12} {model.samples[:, j].mean():>8.4f} {lo:>8.4f} {hi:>8.4f}")

    p_up = model.predict_proba(X_test)
    pred = model.predict(X_test)

    print(f"\nN = {N} days, test rows: {len(y_test)}, share positive: {y_test.mean():.3f}")
    print(f"accuracy:                 {np.mean(pred == y_test):.4f}")
    print(f"baseline (always 'up'):   {max(y_test.mean(), 1 - y_test.mean()):.4f}")
    print(f"ROC AUC:                  {roc_auc_score(y_test, p_up):.4f}")

    # Plot 1: traceplots to check convergence. Plot 2: posterior of each weight
    fig, axes = plt.subplots(2, len(FEATURES), figsize=(16, 6))
    for j, name in enumerate(FEATURES):
        axes[0, j].plot(model.all_samples[:, j])
        axes[0, j].axvline(BURN_IN, color="red", linestyle="--")
        axes[0, j].set_title(f"Trace: {name}")
        axes[1, j].hist(model.samples[:, j], bins=30)
        axes[1, j].axvline(0, color="red", linestyle="--")
        axes[1, j].set_title(f"Posterior: {name}")
    fig.tight_layout()
    plt.show()