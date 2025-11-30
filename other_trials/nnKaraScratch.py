"""
Potentially useless file.
"""


import numpy as np
import pandas as pd

data = pd.read_csv('hp_train.csv')
y = data.SalePrice
data_features = data.columns
features = list()
for i in data_features:
    if data[i].dtype in ['int64', 'float64']:
        if data[i].corr(y) > 0.5:
            # print(f"{i}: {data[i].corr(y)}")
            features.append(i)
data = data[features]
train, test = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]
train_y, test_y = train.SalePrice, test.SalePrice
features.remove('SalePrice')
train_X, test_X = train[features], test[features]
n, m = train_X.shape

print(train_X.shape)


epochs = 300
learning_rate = 0.01

layer1 = np.random.uniform(-0.5, 0.5, (10, 10))
bias1 = np.zeros((20, 1))

layer2 = np.random.uniform(-0.5, 0.5, (10, 1))
bias2 = np.zeros((1, 1))

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# print(train_X)
for epoch in range(epochs):
    for i in range(n):
    
        x = np.array(train_X.iloc[i])
        y = train_y.iloc[i]

        x.shape += (1, )
        print(x.T.shape)

        # forward prop
        z1 = np.dot(x.T, layer1) + bias1
        print(z1 )
        a1 = sigmoid(z1)
        z2 = np.dot(a1, layer2) + bias2
        output = sigmoid(z2)
        print(output)
        # backprop output --> hidden layer (cost function derivative)
        delta = output - y
        print(delta)
        layer2 += -learning_rate * delta * a1
        bias2 += -learning_rate * delta

        # backprop hidden layer --> input layer (activation function derivative)
        delta = np.transpose(layer2) @ delta * (a1 * (1 - a1))
        layer1 += -learning_rate * delta @ np.transpose(train_X.iloc[i])
        bias1 += -learning_rate * delta
        

