from src.models.pretorch import *
import math
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model_Avg():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self):
        self.avg = np.mean(self.y)

    def predict(self):
        # return int(np.round(self.avg))
        return self.avg


class Model_linear1D():
    def __init__(self, x, y):
        self.x = x  # shape (N, 1)
        self.y = y  # shape (N, 1)
        self.a = None
        self.b = None

    def fit(self):
        def lin_func(x, a, b):
            return a * x + b
        popt, pcov = curve_fit(lin_func, self.x, self.y)
        self.a = popt[0]
        self.b = popt[1]
        return self.a, self.b

    def predict(self, x):
        if self.a is None or self.b is None:
            self.fit()
        # return int(np.round(self.a * x + self.b))
        return self.a * x + self.b


class Model_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        y_pred = x.squeeze(1) # (y, 1) -> (y)
        return y_pred


def Model_linear_training(x_data, y_data):
    model = Model_linear().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3)

    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)
    for epoch in range(500):
        optimizer.zero_grad()
        pred_y = model(x_data)
        loss = criterion(pred_y, y_data)
        loss.backward()
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

    return model