#!/usr/bin/env python
#coding: UTF-8

# 每个图片8x8  识别数字：0,1,2,3,4,5,6,7,8,9

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split

# digits = load_digits()
# print(digits.data.shape)
#
# import matplotlib.pylab as pl
# pl.gray()
# pl.matshow(digits.images[6]) #把一个数组或矩阵画成一个图
# pl.show()



digits = load_digits()
X = digits.data
y = digits.target
X -= X.min()  # normalize the values to bring them into the range 0-1
X /= X.max()

nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train) #根据要求把类别标签改成0 1的形式
labels_test = LabelBinarizer().fit_transform(y_test)
print("start fitting")
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
