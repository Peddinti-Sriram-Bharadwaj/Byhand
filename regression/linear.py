import pandas as pd
import numpy as np

def initialize(n_features, initializer = 'zeros' ):
    weights = np.zeros(n_features).reshape(n_features,1)
    print(weights.shape)
    bias = np.array([[0]])
    return (weights, bias)

def predict(x,weights, bias):
    preds = np.dot(x,weights) + bias 
    return preds

def mse(x,y,weights,bias):
    preds = predict(x,y,weights,bias)
    return np.mean((y-preds)**2)

def mse_grad(x,y,preds):
    return np.mean(-2*(y-preds))


def update(x,y,weights,bias, learning_rate):
    preds = predict(x,weights,bias)
    grad = mse_grad(x,y,preds)
    weights = weights - learning_rate*grad*x.T
    bias = bias - learning_rate*grad
    return (weights,bias)

def gradient_descent(x,y,n_features,n_iter,learning_rate):
    weights,bias = initialize(n_features)
    for _ in range(n_iter):
        weights,bias = update(x,y,weights,bias,learning_rate)
    return weights,bias

def linear_fit(x,y, n_iter, learning_rate = 0.01,split = 0.2, loss = 'mse',closed = False):
    #add check for x dims and y dims compatibility
    n_features, n_rows = x.shape[1], x.shape[0]
    train_rows = int(split*n_rows)
    x_train = x[:train_rows]
    y_train = y[:train_rows]
    
    x_test = x[train_rows:]
    y_test = y[train_rows:]

    weights, bias = gradient_descent(x_train,y_train,n_features,n_iter,learning_rate)
    return (weights,bias)

x = np.array([i for i in range(1000)]).reshape(1000,1)
y = np.array([i+3 for i in range(1000)]).reshape(1000,1)

weights,bias = linear_fit(x,y,10)
print(predict(np.array([[1]]),weights,bias))
