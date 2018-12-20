#!/usr/bin/env python
#coding: UTF-8
import numpy as np

#双曲函数
def tanh(x):
    return np.tanh(x)

#双曲函数的导数
def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

#逻辑函数
def logistic(x):
    return 1/(1 + np.exp(-x))

#逻辑函数的导数
def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'): #layers每层神经元的个数。如layers=(10,10,2)表示第一层有10个神经元，第二层有10个，第三层有2个神经元

        """
          :param layers: A list containing the number of units in each layer.
                     Should be at least two values
          :param activation: The activation function to be used. Can be
                     "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        #随机产生权重
        self.weights = []
        for i in range(1, len(layers) - 1):  #除了输入层的第一层开始，分别求当前层和上一层、下一层之间的权重
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25) #表示(layers[i - 1] + 1, layers[i] + 1)大的矩阵，值为-0.25~0.25之间的数
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)



    # X是数据集，每一行代表一个实例，列代表特征值。 y表示类别标签；epochs表示循环次数
    # 采取抽样更新权重
    def fit(self, X, y, learning_rate = 0.2, epochs = 10000):
        # 对X，y进行处理：将X的最后加上一列1；y转换成矩阵形式
        X = np.atleast_2d(X)   #确保X至少是二维的
        temp = np.ones([X.shape[0], X.shape[1] + 1]) #初始化一个矩阵，值全是1。 X如果是10*100的矩阵，那么X.shape有两个值：10和100。因此，这里X.shape[0]取得是数据集的行数。列+1  要放偏差b的
        temp[:, 0:-1] = X  # 除最后一列之外的所有行的列赋值
        X = temp   #此时数据集X相当于增加了一列，值为1，存放的是偏差值b
        y = np.array(y) #将y转为矩阵格式


        for k in range(epochs):
            i = np.random.randint(X.shape[0]) # 从0~X.shape[0] 随机取一行
            a = [X[i]] #第i行实例

            # 前向传播。len(weights)就等于权重矩阵的个数。
            for l in range(len(self.weights)):
                            a.append(self.activation(np.dot(a[l], self.weights[l])))  # Computer the node value for each layer (O_i) using activation function

            #反向传播
            error = y[i] - a[-1]  # Computer the error at the top layer
            deltas = [error * self.activation_deriv(a[-1])]  # For output layer, Err calculation (delta is updated error)

            # Staring backprobagation   len(a)-2:指的是倒数第二层
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):  #反向计算出Err之后，正向更新的W？
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp #以上代码实现将x增加一列。
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a




















