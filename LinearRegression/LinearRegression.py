import numpy as np

#샘플 데이터
x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 12., 15., 18., 21., 24.])

#가중치 초기값
W = 0.0
b = 0.0

#데이터 길이 및 에포크, 학습률 지정
n_data = len(x_train)  
epochs = 500
learning_rate = 0.01


#학습
for i in range(epochs):
    hypothesis = x_train * W + b  
    cost = np.sum((hypothesis - y_train) ** 2) / n_data

    gradient_w = np.sum((W * x_train - y_train + b) * 2 * x_train) / n_data
    gradient_b = np.sum((W * x_train - y_train + b) * 2) / n_data

    W -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    if i % 100 == 0:
        print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, epochs, cost, W, b))

print('W: {:10f}'.format(W))
print('b: {:10f}'.format(b))
print('result : ')
print(x_train * W + b)