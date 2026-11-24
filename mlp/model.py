"""
Code for MLP model, featurization, training, and evaluation
Author: Nick Usich

Goal: Predict 
Model:
Features:
Train:
Eval:
"""
import numpy as np
import torch.nn as nn
import torch.nn.fuctional as F

from loadDataset import load_train, load_test

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y=x
        return y

def featurize():
    return None

def one_epoch():
    return None

def train():
    train = load_train()
    test = load_test()

    x_train, y_train = featurize(train)
    x_test, y_test = featurize(test)

    return None

if __name__=="__main__":
    train()
    