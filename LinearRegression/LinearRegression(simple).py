import numpy as np

from sklearn.datasets import load_diabetes
database = load_diabetes()

#예시 데이터
x = database.data[:,2]
y = database.target

#초기 weight값 설정
def initialization_wb():
    weight = np.random.uniform(low=-1.0, high=1.0)
    bias = np.random.uniform(low=-1.0, high=1.0)
    return weight, bias

#예측 y값 계산 함수 정의
def calculate(weight, x, bias):
    y_hat = weight * x + bias
    return y_hat

#비용 함수 정의
def cost_calculate(y, y_hat):
    cost = ((y - y_hat) ** 2).mean() #MSE
    return cost


def fit(x, y, epoch=10000, lr = 0.5):
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

        if(i + 1) % 1000 == 0:
            print(f"Epoch : {i}, Weight : {weight}, Bias : {bias}, Cost : {cost}")

#run!
fit(x, y, epoch=10000, lr=0.5)

