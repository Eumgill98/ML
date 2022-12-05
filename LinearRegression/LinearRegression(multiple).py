import numpy as np
import itertools
#데이터 불러오기 (보스터 집값)
from sklearn.datasets import load_boston
boston_dataset = load_boston()
x = boston_dataset.data
y = boston_dataset.target
y = y.reshape(len(y),1)

#초기 가중치, 편향 지정
def initialization_wb():
    weight = np.random.uniform(low=-1.0, high=1.0, size=(len(x[0]), 1))
    bias = np.random.uniform(low=-1.0, high=1.0)
    return weight, bias


#비용함수
def cost_calculate(y, y_hat):
    cost = ((y - y_hat) ** 2).mean()
    return cost

#예측값 
def calculate(weight, x, bias):
    y_hat = np.dot(x,weight) + bias
    return y_hat

#학습 진행
def fit(x, y,  epoch=500, lr = 0.0005):
    epoch = epoch
    lr = lr
    weight, bias = initialization_wb()

    for i in range(epoch):
        y_hat = calculate(weight, x, bias)

        cost = cost_calculate(y, y_hat)


        #가중치 및 편향 업데이트
        ##경사하강법 적용
        weight -= lr * ((y_hat - y) * x).mean()
        bias -= lr * (y_hat - y).mean()
        
        print(f"Epoch : {i}, Weight : {weight}, Bias : {bias}, Cost : {cost}")

#run!
fit(x, y, epoch=140, lr=0.0005)

## 학습률이 높거나 에포크가 많으면 무한의 양으로 간다