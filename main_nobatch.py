"""
sgd，无自定义激活函数
"""

import pandas as pd
import numpy as np
from math import ceil
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def _scale_dataset(values, scale_range):
    # normalize features
    scaler = MinMaxScaler(feature_range=scale_range or (0, 1))
    values = scaler.fit_transform(values)
    return values, scaler


# 按照timelength切分
def _series_to_supervised(values, col_names, n_in_timestep, n_out_timestep, dropnan=True, verbose=False):
    n_vars = 1 if type(values) is list else values.shape[1]
    if col_names is None:
        col_names = ["var%d" % (j + 1) for j in range(n_vars)]
    df = pd.DataFrame(values)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in_timestep, 0, -1):
        cols.append(df.shift(i))
        names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out_timestep):
        cols.append(df.shift(-i))  # 这里循环结束后cols是个列表，每个列表都是一个shift过的矩阵
        if i == 0:
            names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
        else:
            names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    if verbose:
        print("\nsupervised data shape:", agg.shape)
    return agg


# 切分数据集
def _train_test_split(values, train_test_split, target_col_index, n_in_timestep, n_features, verbose=0):
    n_train_count = ceil(values.shape[0] * train_test_split)
    train = values[:n_train_count, :]
    test = values[n_train_count:, :]

    # split into input and outputs
    n_obs = n_in_timestep * n_features
    train_x, train_y = train[:, :n_obs], train[:, target_col_index]

    test_x, test_y = test[:, :n_obs], test[:, target_col_index]

    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], n_in_timestep, n_features))
    test_x = test_x.reshape((test_x.shape[0], n_in_timestep, n_features))

    if verbose:
        print("")
        print("train_X shape:", train_x.shape)
        print("train_y shape:", train_y.shape)
        print("test_X shape:", test_x.shape)
        print("test_y shape:", test_y.shape)
    return train_x, train_y, test_x, test_y


def load_data(file_path, target_col_name, input_dim, output_dim):
    data = pd.read_csv(file_path, encoding='gbk')
    data.drop('date', axis=1, inplace=True)

    col_names = list(data.columns)
    n_features = len(col_names)
    target_col_index = col_names.index(target_col_name) - n_features

    values, scaler = _scale_dataset(data.values, None)
    data_new = _series_to_supervised(values, col_names=col_names, n_in_timestep=input_dim, n_out_timestep=output_dim)

    # 切分训练集测试集
    trainX, trainy, testX, testy = _train_test_split(values=data_new.values, train_test_split=0.7,
                                                     target_col_index=target_col_index, n_in_timestep=input_dim,
                                                     n_features=n_features)

    return trainX, trainy, testX, testy, scaler


class RNN_NUMPY():
    def __init__(self, hidden_dim=10, learning_rate=0.001, num_iter=100, batch_size=10):
        # 超参数
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.batch_size = batch_size

        # 模型系数
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.output_dim = None
        self.paras = dict()
        self.lst_loss = []

    def init_paras(self, X, y):
        self.input_dim = X.shape[-1]
        self.output_dim = 1 if len(y.shape) == 1 else y.shape[1]
        self.paras['U'] = np.random.uniform(-np.sqrt(1. / self.input_dim), np.sqrt(1. / self.input_dim),
                                            (self.hidden_dim, self.input_dim))
        self.paras['V'] = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim),
                                            (self.output_dim, self.hidden_dim))
        self.paras['W'] = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim),
                                            (self.hidden_dim, self.hidden_dim))
        self.paras['ba'] = np.zeros((self.hidden_dim, 1))
        self.paras['by'] = np.zeros((self.output_dim, 1))

    def forward(self, xi):
        a = list()
        a.append(np.zeros((self.hidden_dim, 1)))
        for t in range(xi.shape[0]):
            a_next = np.tanh(np.dot(self.paras['U'], xi[[t]].T) + np.dot(self.paras['W'], a[-1]) + self.paras['ba'])
            a.append(a_next)
        yt = np.dot(self.paras['V'], a[-1]) + self.paras['by']
        return a, yt

    def backward(self, xi, ai, yi, yi_pred):
        # 初始化梯度
        d_paras = dict()
        d_paras['U'] = np.zeros_like(self.paras['U'])
        d_paras['W'] = np.zeros_like(self.paras['W'])
        d_paras['V'] = np.outer(2 * (yi_pred - yi), ai[-1].T)

        da = np.dot(self.paras['V'].T, 2 * (yi_pred - yi))
        dzt = da * (1 - np.tanh(ai[-1]) ** 2)

        # 时间序列回溯
        for t in range(xi.shape[0])[::-1]:
            d_paras['W'] += np.dot(dzt, ai[t - 1].T)
            d_paras['U'] += np.dot(dzt, xi[[t - 1]])

            # 更新dzt :=  dz(t-1)
            dzt = np.dot(self.paras['W'].T, dzt) * (1 - ai[t - 1] ** 2)

        return d_paras

    def cal_loss(self, y_true, y_pred):
        return np.square(y_true - y_pred)

    def update_paras(self, d_paras):
        for i in d_paras.keys():
            self.paras[i] -= self.learning_rate * d_paras[i]

    def fit(self, X, y):
        # 初始化各参数
        self.init_paras(X, y)

        # 根据batch计算需要迭代次数
        # 建模
        for iter in range(self.num_iter):
            iter_loss = 0

            for i in range(X.shape[0]):
                # 前向传播
                ai, yit = self.forward(X[i])

                # 计算loss
                iter_loss += self.cal_loss(y[i], yit)

                # 后向传播
                d_paras = self.backward(X[i], ai, y[i], yit)

                # 更新参数
                self.update_paras(d_paras)
            self.lst_loss.append(iter_loss)
            print('[iter %d]: loss:%.4f' % (iter, iter_loss))

    def predict(self, X, y, scaler):
        y_pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            _, y_pred[i] = self.forward(X[i])

        tmp = X[:, -1].copy()
        tmp[:, -1] = y_pred[:, 0]
        inv_yhat_t1 = scaler.inverse_transform(tmp)
        inv_yhat_t1 = inv_yhat_t1[:, -1]

        # invert scaling for actual, t+1天的销量
        tmp = X[:, -1].copy()
        tmp[:, -1] = y
        inv_y_t1 = scaler.inverse_transform(tmp)
        inv_y_t1 = inv_y_t1[:, -1]
        return inv_y_t1, inv_yhat_t1


if __name__ == '__main__':
    # 参数设置
    n_in_timestep = 3
    n_out_timestep = 1

    # 数据导入
    trainX, trainy, testX, testy, scaler = load_data('./data/stock_dataset_2.csv', 'high', n_in_timestep,
                                                     n_out_timestep)

    # 建模
    model = RNN_NUMPY(hidden_dim=10, learning_rate=0.001, num_iter=10, batch_size=10)
    model.fit(trainX, trainy)
    testy_true, testy_pred = model.predict(testX, testy, scaler)

    # 绘图
    plt.plot(testy_true, label='Actual')
    plt.plot(testy_pred, label='Predict')
    plt.legend()
    plt.show()
