import numpy as np
import itertools

#data_load
from sklearn.datasets import load_boston
boston_dataset = load_boston()
x = boston_dataset.data
y = boston_dataset.target
y = y.reshape(len(y),1)

#초기 가중치, 편향 지정(init_w_b)
def initialization_wb():
    weight = np.random.uniform(size=(len(x[0]), 1))
    bias = np.random.uniform()
    return weight, bias

#비용함수(cost)
def cost_calculate(y, y_hat):
    cost = ((y - y_hat) ** 2).mean()
    return cost

#예측값(predict_y)
def calculate(weight, x, bias):
    y_hat = np.dot(x,weight) + bias
    return y_hat

#학습 진행(run, train)
def fit(x, y,  epoch=500, lr = 0.0005):
    weight, bias = initialization_wb()

    #매 epoch마다 랜덤한 x,y 값 가져오기
    
    for i in range(epoch):
        rand_x  = np.random.randint(low=len(x), size=1)
        x_sgd = x[rand_x[0]]
        y_sgd =  y[rand_x[0]]

        #예측값 계산
        y_hat = calculate(weight, x_sgd, bias)

        #가중치 업데이트(확률적 경사하강법)
        weight = weight -  lr*(2*(y_hat - y_sgd) * x_sgd).mean()
        bias = bias - lr*(2*(y_hat - y_sgd)).mean()

        #cost 
        y_hat = calculate(weight, x, bias)
        cost = cost_calculate(y, y_hat)

 
        print(f"Epoch : {i}, Weight : {weight}, Bias : {bias}, Cost : {cost}")

#run!
fit(x, y, epoch=100, lr=0.000005)



