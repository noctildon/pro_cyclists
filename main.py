import sys
import os
import numpy as np
from src.features.build_features import rider_features, riders_num
from src.visualization.visualize import plot_results, stats_results
from src.models.train_model import simple_models_training, Train_pl
import multiprocess as mp
from filelock import FileLock
global lock, outputFile


def get_rider_data(rider_id):
    rider = rider_features(rider_id)
    if rider is None: return None
    rider = rider.astype(float)
    xx = rider[:, :-1] # date, distance
    yy = rider[:, -1] # ranking
    return xx, yy


def Simple_model(xx, yy):
    return simple_models_training(xx, yy, ratio=0.7, verbose=False)


def NN_Model_pl(xx, yy):
    data_config = {
        'xx': xx,
        'yy': yy,
        'batch_size': 1,
        'valid_ratio': 0.3,
        'verbose': False,
    }
    model_config = {
        'model_type': 'DNN',
        'n_epochs': 40,
        'lr': 5e-2,
        'patience': 20,
        'save_path': 'models',
        'save_name': '{epoch:d}',
        'tb_logs': False
    }
    train_pl = Train_pl(data_config, model_config)
    best_valid_loss = train_pl.train(show_progressbar=True)
    return best_valid_loss


def LSTM_Model_pl(xx, yy):
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
        'n_epochs': 40,
        'lr': 5e-2,
        'num_layers': 8,
        'hidden_size': 2,
        'dropout': 0.2,
        'patience': 20,
        'save_path': 'models',
        'save_name': '{epoch:d}',
        'tb_logs': False
    }
    train_pl = Train_pl(data_config, model_config)
    best_valid_loss = train_pl.train(show_progressbar=True)
    return best_valid_loss


def testing_models():
    data = get_rider_data(5)
    xx, yy = data
    r = NN_Model_pl(xx, yy)
    print('NN Model', r)

    # r = LSTM_Model_pl(xx, yy)
    # print('LSTM Model', r)


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


if __name__ == "__main__":
    lock = FileLock(outputFile + '.lock')

    testing_models()
    # run(parallel=True)
    # read_losses()

    # plot_results()
    # stats_results()
