from torch.utils.data import Dataset
import torch
import numpy as np


class RaceDataset(Dataset):
    '''
    x: Features (date, distance)
    y: Ranking, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


def preprocess(xx, yy):
    # date
    xx[:, 0] = xx[:, 0] - xx[0,0]
    xx[:, 0] /= max(xx[:, 0])

    # distance
    xx[:, 1] /= max(xx[:, 1])

    # ranking
    yy /=  max(yy)
    return xx, yy