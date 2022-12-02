import numpy as np

from sklearn.datasets import load_diabetes
database = load_diabetes()

#예시 데이터
x = database.data[:,2]
y = database.target


def initialization_wb():
    weight = np.random.uniform(low=-1.0, high=1.0)
    bias = np.random.uniform(low=-1.0, high=1.0)
    return weight, bias


def cost_calculate(y, y_hat):
    cost = ((y - y_hat) ** 2).mean()
    return cost


def calculate(weight, x, bias):
    y_hat = weight * x + bias
    return y_hat


def fit(x, y, epoch=10000, lr = 0.5):

    weight, bias = initialization_wb()

    for i in range(epoch):
        #매번 랜덤값 가져오기
        x_sgd = np.random.choice(x)
        y_sgd =  y[np.where(x== x_sgd)[0][0]]

        #예측값 계산
        y_hat = calculate(weight, x_sgd, bias)

        #가중치 업데이트(확률적 경사하강법)
        weight -= lr*((y_hat - y_sgd) * x_sgd)
        bias -= lr*(y_hat - y_sgd)

        #cost 
        predict = weight * x + bias
        cost = cost_calculate(y, predict)

        if(i + 1) % 1000 == 0:
            print(f"Epoch : {i}, Weight : {weight}, Bias : {bias}, Cost : {cost}")

#run!
fit(x, y, epoch=10000, lr = 0.5)
