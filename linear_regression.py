# -*- coding: utf-8 -*-
"""linear_regression
# this is linear regression implementation in numpy and pytorch
# for simplicity, this is only single variable regression
"""

import torch
import torch.nn as nn
import numpy as np
import random

class LinearRegression(object):
    def __init__(self, implementation='numpy'):
        self.implementation= implementation
    def fit(self, x ,y):
        if self.implementation == 'numpy':
            self._fit_by_numpy(x,y)
        elif self.implementation == 'pytorch':
            self._fit_by_pytorch(x,y)

    def _fit_by_pytorch(self, x, y):
        X_test = torch.from_numpy(x.astype(np.float32))
        y_test = torch.from_numpy(y.reshape(150,1).astype(np.float32))
        # define linear regression  model
        n_input, n_out, batch_size, learning_rate = 1, 1, 100, 0.1
        self.model = nn.Sequential(nn.Linear(n_input, n_out ))

        # define loss function and optimzer on how to upgrade the gradient
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # actual training is happening
        losses = []
        for epoch in range(10000):
            pred_y = self.model(X_test[:,:])
            loss = loss_function(y_test, pred_y)
            losses.append(loss.item())

            self.model.zero_grad()
            loss.backward()
            optimizer.step()
        # weight and bias (w,b)
        for name, param in self.model.named_parameters():
            print("fit_by_pytorch, weight and bias: ",name, param)


    def _fit_by_numpy(self, x, y):
        #idx = random.randint(0, 149)
        self.w=11
        self.b=1
        self.lr=0.1
        for i in range(10000):
            idx = random.randint(0, 149)
            pred= self.w*x[idx]+self.b
            self.w= self.w-(pred-y[idx])*x[idx]
            self.b= self.b-(pred-y[idx])*1
        print("fit_by_numpy, weight and bias: ",self.w,self.b)

    def _predict_by_pytorch(self, x):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)
        return prediction
    def _predict_by_numpy(self, x):
        prediction= self.w*x+self.b
        return prediction

    def predict(self, x, implementation):
        if implementation == 'numpy':
            return self._predict_by_numpy(x)
        elif implementation == 'pytorch':
            return self._predict_by_pytorch(x)

# sample use
if __name__ == '__main__':
    from sklearn.datasets import make_regression
    X_test, y_test, coef = make_regression(n_samples=150, n_features=1, noise=6, random_state=42, coef=True, bias=1)
    print("ground truth weight is: ", coef)
    X_test = X_test.reshape(150,1)
    lr=LinearRegression(implementation='pytorch')
    lr.fit(X_test,y_test)
    a = np.array([10])
    a= torch.from_numpy(a.astype(np.float32))

    print(lr.predict(a, 'pytorch'))
    ##
    lr=LinearRegression(implementation='numpy')
    lr.fit(X_test,y_test)
    print(lr.predict(10, 'numpy'))

