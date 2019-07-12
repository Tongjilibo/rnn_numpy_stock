# rnn_numpy_stock
## 1.0版本
1. 为sgd随机梯度下降，没有batch概念
2. 没有更新常数项参数ba和by

## 1.1版本

### Already done

1. 增加了batch_size，可以做Mini-batch梯度下降
2. 在backward中更新常数项参数
3. 支持自定义激活函数

### To do
1. dropout, adam, 正则化
2. 多层rnn