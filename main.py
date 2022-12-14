import sys
import os
import numpy as np
from src.features.build_features import rider_features, riders_num
from src.models.train_model import simple_models_training, Train_pl
import multiprocess as mp
from filelock import FileLock
global lock, outputFile


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
    return simple_models_training(xx, yy, ratio=0.7, verbose=False)


def NN_Model_pl(xx, yy):
    # best loss: 0.080
    data_config = {
        'xx': xx,
        'yy': yy,
        'batch_size': 1,
        'valid_ratio': 0.3,
        'verbose': False,
    }
    model_config = {
        'model_type': 'DNN',
        'n_epochs': 2000, # 2000
        'lr': 1e-5,
        'patience': 600,
        'save_path': 'models',
        'save_name': '{epoch:d}',
        'tb_logs': False
    }
    train_pl = Train_pl(data_config, model_config)
    best_valid_loss = train_pl.train(show_progressbar=False)
    return best_valid_loss


def LSTM_Model_pl(xx, yy):
    # best loss: 0.072
    data_config = {
        'xx': xx,
        'yy': yy,
        'batch_size': 1,
        'valid_ratio': 0.3,
        'verbose': False,
    }
    model_config = {
        'model_type': 'LSTM',
        'input_size': 2,
        'n_epochs': 2000,  # 2000
        'lr': 1e-5,
        'num_layers': 8,
        'hidden_size': 2,
        'dropout': 0.2,
        'patience': 600,
        'save_path': 'models',
        'save_name': '{epoch:d}',
        'tb_logs': False
    }
    train_pl = Train_pl(data_config, model_config)
    best_valid_loss = train_pl.train(show_progressbar=False)
    return best_valid_loss


def testing_models():
    data = get_rider_data(5)
    xx, yy = data
    # r = NN_Model_pl(xx, yy)
    r = LSTM_Model_pl(xx, yy)
    print(r)


outputFile = 'models/results.txt'
def train_all_models(i, lock=None):
    data = get_rider_data(i)
    if data is None:
        return
    xx, yy = data
    res = [i]
    res += list(Simple_model(xx, yy))
    res.append(NN_Model_pl(xx, yy))
    res.append(LSTM_Model_pl(xx, yy))

    print('res', res)
    if lock:
        with lock:
            with open(outputFile, "a") as f:
                np.savetxt(f, res, newline=' ')
                f.write('\n')
    else:
        with open(outputFile, "a") as f:
            np.savetxt(f, res, newline=' ')
            f.write('\n')


def run(cores=6, parallel=False):
    if parallel:
        mp.set_start_method('spawn')
        pool = mp.Pool(processes=cores)
        pool.map(train_all_models, range(riders_num))

    else:
        for i in range(riders_num):
            train_all_models(i)


def read_losses():
    f = np.genfromtxt(outputFile, delimiter=' ')
    print(f.shape)


if __name__ == "__main__":
    lock = FileLock(outputFile + '.lock')

    # testing_models()
    run(parallel=True)
    # read_losses()
