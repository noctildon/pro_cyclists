from src.models.pretorch import *
from scipy.optimize import curve_fit
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn


class Model_Avg():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self):
        self.avg = np.mean(self.y)

    def predict(self):
        return self.avg


class Model_linear_date():
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
        return self.a * x + self.b


class Model_XBG():
    def __init__(self, n_estimators=5, max_depth=6, eta=0.05):
        self.model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=max_depth, n_estimators=n_estimators, eta=eta)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class Model_LGB():
    def __init__(self, n_estimators=16, max_depth=6, learning_rate=1e-2):
        self.model = lgb.LGBMRegressor(
            objective="regression", max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class Model_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        y_pred = x.squeeze(1) # (y, 1) -> (y)
        return y_pred


class Model_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        x = self.linear(x)
        y_pred = x.squeeze(1) # (y, 1) -> (y)
        return y_pred


class Model_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, bidirectional=False):
        super().__init__()
        self.bid = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(self.bid*hidden_size, 1)

    def forward(self, x):
        h0 = torch.randn(self.bid*self.num_layers, self.hidden_size).to(device)
        c0 = torch.randn(self.bid*self.num_layers, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0)) # (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = out.squeeze(1) # (y, 1) -> (y)
        return out
