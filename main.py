import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.features.build_features import rider_features, riders_num
from src.models.train_model import Model_NN_training, simple_models_training


# get the rider race data
rider = rider_features(30).astype(float)
xx = rider[:, :-1] # date, distance
yy = rider[:, -1] # ranking


def NN_model(xx, yy):
    config = {
        'ratio': 0.7,
        'learning_rate': 5e-4,
        'n_epochs': 20000,
        'batch_size': 32,
        'save_path': 'models/model.pth',
        'early_stop': 600
    }
    Model_NN_training(xx, yy, config)


if __name__ == "__main__":
    # simple_models_training(xx, yy)
    NN_model(xx, yy)