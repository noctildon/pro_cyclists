import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.pretorch import *
from src.features.build_features import rider_features, riders_num
from src.models.train_model import Model_NN_training, simple_models_training, Model_LSTM_training


def get_rider_data(rider_id):
    rider = rider_features(rider_id)
    if rider is None:
        return None
    rider = rider.astype(float)
    xx = rider[:, :-1] # date, distance
    yy = rider[:, -1] # ranking
    return xx, yy


def Simple_model(xx, yy):
    # Average model loss: 0.09
    # Linear model loss: 0.094
    return simple_models_training(xx, yy, ratio=0.7)


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
    return Model_NN_training(xx, yy, config)


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
    return Model_LSTM_training(xx, yy, config)


def train_all_models(i):
    data = get_rider_data(i)
    if data is None:
        return
    xx, yy = data
    res = [i]
    res += list(Simple_model(xx, yy))
    res.append(NN_model(xx, yy))
    res.append(LSTM_model(xx, yy))

    # res = np.array(res)
    print('res', res)
    with open('models/results.txt', "a") as f:
        np.savetxt(f, res, newline=' ')
        f.write('\n')


if __name__ == "__main__":
    for i in range(riders_num):
        train_all_models(i)
