from src.models.preprocessing import *
from src.models.models import *
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import logging


# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
accelerator, devices = ("gpu", 1) if torch.cuda.is_available() else (None, None)


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


    # lightgbm
    lgb_model = Model_LGB()
    lgb_model.fit(x_train, y_train)
    y_pred_lgb = lgb_model.predict(x_valid)
    mse_lgb = np.mean((y_pred_lgb - y_valid) ** 2)
    print(f'LGB MSE: {mse_lgb}')

    return mse_avg, mse_1D, mse_xgb, mse_lgb


class RaceDataModule(pl.LightningDataModule):
    def __init__(self, xx, yy, batch_size=64, valid_ratio=0.3, num_workers=4, **kwargs):
        super().__init__()
        self.xx = xx
        self.yy = yy
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.num_workers = num_workers

    def setup(self, stage):
        # the ratio in train_valid_split is train ratio, ie. 1-valid_ratio
        x_train, y_train, x_valid, y_valid = train_valid_split(self.xx, self.yy, ratio=1-self.valid_ratio)
        train_dataset = RaceDataset(x_train, y_train)
        valid_dataset = RaceDataset(x_valid, y_valid)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)


class Model_pl(pl.LightningModule):
    def __init__(self, model_type, valid_size, lr=5e-4, **kwargs):
        super().__init__()
        self.lr = lr
        self.criterion = nn.MSELoss(reduction='mean')
        self.model_type = model_type
        self.best_valid_loss = np.inf
        self.valid_size = valid_size

        if model_type == 'DNN':
            self.model = Model_DNN(**kwargs)
        elif model_type == 'LSTM':
            self.model = Model_LSTM(**kwargs, device=self.device)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log_dict({'val_loss': loss}, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        if len(outputs) < self.valid_size:
            return
        avg_valid_loss = torch.stack(outputs).mean()
        if avg_valid_loss < self.best_valid_loss:
            self.best_valid_loss = avg_valid_loss


class Train_pl():
    def __init__(self, data_config, model_config):
        for key, value in (data_config | model_config).items():
            setattr(self, key, value)

        # the valid batch size
        model_config['valid_size'] = int(data_config['xx'].shape[0] * data_config['valid_ratio'])
        self.data_config = data_config
        self.model_config = model_config
        self.data_model_setup()
        self.callbacks_setup()

    def data_model_setup(self):
        self.Data = RaceDataModule(**self.data_config)
        self.Model = Model_pl(**self.model_config)

    def callbacks_setup(self):
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename=self.save_name, dirpath=self.save_path, save_top_k=0)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        bar = RichProgressBar(leave=True, theme=RichProgressBarTheme(
                description='green_yellow', progress_bar='green1', progress_bar_finished='green1'))
        self.callbacks = [checkpoint_callback, early_stopping, bar]

    def train(self):
        trainer = pl.Trainer(accelerator=accelerator, devices=devices, callbacks=self.callbacks,
                        logger=self.tb_logs, max_epochs=self.n_epochs)
        trainer.fit(self.Model, self.Data)
        return self.Model.best_valid_loss.cpu().detach().numpy()
