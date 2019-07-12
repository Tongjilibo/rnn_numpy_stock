import numpy as np


# sigmod 激活函数
def sigmod(z):
    return 1/(1+np.exp(-z))


# sigmod梯度
def d_sigmod(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)


# relu激活函数
def relu(z):
    return np.where(z < 0, 0, z)


# relu梯度
def d_relu(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


# tanh梯度
def d_tanh(z):
    return 1 - np.tanh(z)**2


# linear激活函数
def linear(z):
    return z


# linear梯度
def d_linear(z):
    return 1


# loss为mse
def mse(y_true, y_pred):
    return np.square(y_pred - y_true).sum()


# loss为mse的梯度
def d_mse(y_true, y_pred):
    return 2 * (y_pred-y_true)


dict_active_fun = {'sigmod': (sigmod, d_sigmod),
                   'tanh': (np.tanh, d_tanh),
                   'relu': (relu, d_relu),
                   'linear': (linear, d_linear)}

dict_loss_fun = {'mse': (mse, d_mse)}
