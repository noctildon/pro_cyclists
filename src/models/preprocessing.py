import numpy as np


def preprocess(xx, yy):
    # date
    xx[:, 0] = xx[:, 0] - xx[0,0]
    xx[:, 0] /= max(xx[:, 0])

    # distance
    xx[:, 1] /= max(xx[:, 1])

    # ranking
    yy /=  max(yy)
    return xx, yy