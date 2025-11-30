def main():
    # Reading and preprocessing some parts of the data.
    data = pd.read_csv('hp_train.csv')
    y = data.SalePrice
    data_features = data.columns
    features = list()
    for i in data_features:
        if data[i].dtype in ['int64', 'float64']:
            if data[i].corr(y) > 0.5:
                features.append(i)
    data = data[features]
    data = data.sample(frac=1).reset_index(drop=True)

    train, test = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]
    train_y, test_y = train.SalePrice, test.SalePrice

    max_price_train = np.max(train_y)
    max_price_test = np.max(test_y)

    train_y = normalize(train_y / max_price_train)
    test_y = normalize(test_y / max_price_test)

    features.remove('SalePrice')
    train_X, test_X = train[features], test[features]
    print(len(train_X.loc[0]))
    #
    network = [
        Dense(10, 8),
        Tanh(),
        Dense(8, 1),
        Tanh()
    ]
    # 
    epochs = 10
    learning_rate = 0.001
    # training
    for e in range(epochs):
        error = 0
        iter = 0
        for i in range(len(train_X)):
            iter += 1
            # forward
            x = train_X.iloc[i]
            output = np.array([x]).T
            ln = 0
            for layer in network:
                ln += 1
                output = layer.forward(output)
            y = np.array(train_y.iloc[i])
            # error
            error += mse(y, output)
            # backward
            grad = mse_prime(y, output)
            for layer in reversed(network):
                ln -= 1
                grad = layer.backward(grad, learning_rate)

        print('\n\n\n\n%d/%d, error=%f' % (e + 1, epochs, error))
    print("FINISHED TRAINING. STARTING TESTING")

    # testing
    testing_error = 0
    for i in range(len(test_X)):
        x = test_X.iloc[i]
        output = np.array([x]).T
        for layer in network:
            output = layer.forward(output)
        y_true = np.array(test_y.iloc[i])
        testing_error += mse(y_true, output)

    print('Total testing error: ', testing_error)
    
import numpy as np
import pandas as pd
from nn import *





main()

