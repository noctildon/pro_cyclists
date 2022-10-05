from src.models.pretorch import *
from src.models.preprocessing import *
import math
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
from tqdm import tqdm
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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


def simple_models_training(xx, yy, ratio=0.7):
    x_train, y_train, x_valid, y_valid = train_valid_split(xx, yy, ratio=ratio)

    # Average model
    model_avg = Model_Avg(x_train, y_train)
    model_avg.fit()
    y_pred_avg = model_avg.predict()
    mse_avg = np.mean((y_pred_avg - y_valid) ** 2)
    print(f'Avg MSE: {mse_avg}')


    # 1D Linear model (x=date, y=ranking)
    model_lin = Model_linear_date(x_train[:, 0], y_train)
    model_lin.fit()
    y_pred_lin = []
    for x in x_valid[:, 0]:
        y_pred_lin.append(model_lin.predict(x))
    y_pred_lin = np.array(y_pred_lin)
    mse_1D = np.mean((y_pred_lin - y_valid) ** 2)
    print(f'1D MSE: {mse_1D}')


    # xgboost
    xgb_model = Model_XBG()
    xgb_model.fit(x_train, y_train)
    y_pred_xgb = xgb_model.predict(x_valid)
    mse_xgb = np.mean((y_pred_xgb - y_valid) ** 2)
    print(f'XGB MSE: {mse_xgb}')


class Model_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        y_pred = x.squeeze(1) # (y, 1) -> (y)
        return y_pred


def Model_NN_training(xx, yy, config):
    model = Model_NN().to(device)

    criterion = nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-5)

    x_train, y_train, x_valid, y_valid = train_valid_split(xx, yy, ratio=config['ratio'])
    train_dataset = RaceDataset(x_train, y_train)
    valid_dataset = RaceDataset(x_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    n_epochs, best_loss = config['n_epochs'], math.inf
    step = early_stop_count = 0

    logdir = f'models/runs/NN/epochs={n_epochs}_lr={config["learning_rate"]}'
    writer = SummaryWriter(log_dir=logdir)

    for epoch in range(n_epochs):
        ### Training ###
        model.train()
        train_loss_record = []
        train_pbar = tqdm(train_loader)
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            train_loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(train_loss_record)/len(train_loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        ### Validation ###
        model.eval()
        valid_loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)

                loss = criterion(pred, y)

            valid_loss_record.append(loss.item())

        mean_valid_loss = sum(valid_loss_record)/len(valid_loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


class Model_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.randn(self.num_layers, self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0)) # (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = out.squeeze(1) # (y, 1) -> (y)
        return out


def Model_LSTM_training(xx, yy, config):
    input_size = 2  # number of features (columns)
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    model = Model_LSTM(input_size, hidden_size, num_layers).to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    x_train, y_train, x_valid, y_valid = train_valid_split(xx, yy, ratio=config['ratio'])
    train_dataset = RaceDataset(x_train, y_train)
    valid_dataset = RaceDataset(x_valid, y_valid)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    n_epochs, best_loss = config['n_epochs'], math.inf
    for epoch in range(n_epochs):
        train_pbar = tqdm(train_loader)

        ### Training ###
        model.train()
        train_loss_record = []

        for x, y in train_pbar:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            scores = model(x)
            loss = criterion(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_record.append(loss.detach().item())
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        mean_train_loss = sum(train_loss_record)/len(train_loss_record)

        ### Validation ###
        model.eval()
        valid_loss_record = []
        for x, y in valid_loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            with torch.no_grad():
                scores = model(x)
                loss = criterion(scores, y)

            valid_loss_record.append(loss.item())
        mean_valid_loss = sum(valid_loss_record)/len(valid_loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')


        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
