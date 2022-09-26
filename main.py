import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.features.build_features import rider_features, riders_num
from src.models.train_model import Model_Avg, Model_linear1D, Model_linear_training
from src.models.preprocessing import preprocess


# get the rider race data
rider = rider_features(30).astype(float)
xx = rider[:, :-1] # date, distance
yy = rider[:, -1] # ranking

# normalize data
xx, yy = preprocess(xx, yy)


# Linear model
lin_model = Model_linear_training(xx, yy)
yhat_linear = lin_model(torch.tensor(xx, dtype=torch.float32))
print(f'yhat_linear: {yhat_linear}')


# Average model
model_avg = Model_Avg(xx, yy)
model_avg.fit()
yhat_avg = model_avg.predict()
print(f'avg prediction: {yhat_avg}')


# 1D Linear model (x=date, y=ranking)
model_lin = Model_linear1D(xx[:, 0], yy)
model_lin.fit()
yhat_1D = []
for x in xx[:, 0]:
    yhat_1D.append(model_lin.predict(x))
yhat_1D = np.array(yhat_1D)


# Plot the data
plt.scatter(xx[:, 0], yy, label='data')
plt.plot(xx[:, 0], yhat_avg*np.ones_like(xx[:, 0]), 'r', label='avg')
plt.plot(xx[:, 0], yhat_1D, 'b', label='1D linear model')
plt.plot(xx[:, 0], yhat_linear.detach().numpy(), 'g', label='linear model')
plt.legend()
plt.yscale('log')
plt.show()