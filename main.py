import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.pretorch import *
from src.features.build_features import rider_features, riders_num
from src.models.train_model import Model_NN_training, simple_models_training, Model_LSTM_training


# get the rider race data
rider = rider_features(30).astype(float)
xx = rider[:, :-1] # date, distance
yy = rider[:, -1] # ranking


def Simple_model(xx, yy):
    # Average model loss: 0.09
    # Linear model loss: 0.094
    simple_models_training(xx, yy, ratio=0.7)


def NN_model(xx, yy):
    config = {
        'ratio': 0.7,
        'learning_rate': 5e-4,
        'n_epochs': 2000,
        'batch_size': 1,
        'save_path': 'models/model.pth',
        'early_stop': 600
    }
    # best loss: 0.080
    Model_NN_training(xx, yy, config)


def LSTM_model(xx, yy):
    config = {
        'num_layers': 8,
        'hidden_size': 2,
        'learning_rate': 5e-4,
        'dropout': 0.2,
        'early_stop': 600,
        'batch_size': 1,
        'ratio': 0.7,
        'n_epochs': 2000,
        'save_path': 'models/model.pth',
    }
    # best loss: 0.072
    Model_LSTM_training(xx, yy, config)


if __name__ == "__main__":
    # Simple_model(xx, yy)
    # NN_model(xx, yy)
    LSTM_model(xx, yy)
