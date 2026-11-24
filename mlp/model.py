"""
Code for MLP model, featurization, training, and evaluation
Author: Nick Usich

Goal: based on weekly prices, can the MLP predict whether a stock declines 5% in the next 60 days?
Model: MLP
Features: center/scale to mean of zero and variance of 1. Binary label: beat/not beat
Train: 500 samples
Eval: 50 samples
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loadDataset import load_train, load_test

class MLP(nn.Module):
    def __init__(self, input_features=2148, output_features=1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features=int(input_features), out_features=input_features//2), nn.ReLU(), 
                                 nn.Linear(in_features=int(input_features//2), out_features=int(input_features//4)), nn.ReLU(), 
                                 nn.Linear(in_features=int(input_features//4), out_features=int(input_features//8)), nn.ReLU(), 
                                 nn.Linear(in_features=int(input_features//8), out_features=int(output_features)))

    def forward(self, x):
        y = self.mlp(x)
        return y

def featurize(split, test_days=60, price_increase_percent=-0.05):
    arrys = []
    for _, arry in split.items():
        arrys.append(arry)
    feats = np.array(arrys)
    n_days = 2147 + 365 - test_days
    y = feats[:, n_days:]
    prices = feats[:, :n_days+1]
    weekly = prices[:, ::5]
    x = np.diff(np.log(weekly), axis=1)

    # label: was this stock better than avg S&P500 year over year? (assuming generous 10% yearly return for S&P) 
    y = np.where((y[:, -1] - y[:, 0])/y[:, 0] > price_increase_percent, 1, 0)
    print("positive-to-total ratio (50=good):", np.sum(y)/y.shape[0])

    print("x", x.shape)
    print("y", y.shape)
    return x, y


def one_epoch(model, x, y, optim):
    pred = model.forward(x).squeeze()

    optim.zero_grad()
    loss = F.binary_cross_entropy_with_logits(pred, y)
    loss.backward()
    optim.step()

def Train(x_train_tr, y_train, epochs=1000):

    x_train_tr, y_train = torch.tensor(x_train_tr, dtype=torch.float32), torch.tensor(y_train,  dtype=torch.float32)

    model = MLP(input_features=x_train_tr.shape[1])
    optim = Adam(params=model.parameters(),
                 lr=1e-3,
                 weight_decay=1e-4)

    for epoch in range(epochs):
        one_epoch(model, x_train_tr, y_train, optim)
    return model

if __name__=="__main__":
    train = load_train()
    test = load_test()

    x_train, y_train = featurize(train)
    x_test, y_test = featurize(test)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_tr = scaler.transform(x_train)
    x_test_tr = scaler.transform(x_test)
    x_test_tr = torch.tensor(x_test_tr,  dtype=torch.float32)

    m = Train(x_train_tr, y_train)
    
    with torch.no_grad():
        m.eval()
        y_test_pred = F.sigmoid(m.forward(x_test_tr)).round().numpy()
        y_test_scores = F.sigmoid(m.forward(x_test_tr)).numpy() 
        mtrx = confusion_matrix(y_test, y_test_pred)
        roc = roc_auc_score(y_score=y_test_scores, y_true=y_test)
        print(mtrx)
        print("roc: ", roc)