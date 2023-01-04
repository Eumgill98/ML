from sklearn import tree
import pandas as pd
import numpy as np
import math

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X_train = data.data
y_train = np.where(data.target == 0, 1, -1)


def AdaBoost(X_train, y_train, n_estimators, learning_rate):
    #분류기를 닮을 리스트 생성
    classifiers = []
    
    #가중치 초기화(데이터 수만큼 가중치 생성)
    N = len(y_train)
    w_i = np.array([1 / N] * N)


    #T = 사용한 기본 모델 갯수
    T = n_estimators
    
    #각 모델의 에러를 저장할 리스트
    clf_erros = []


    #각 모델 마다 학습
    for t in range(T):
        clf = tree.DecisionTreeClassifier(max_depth = 1) #깊이가 1인 의사결정나무 - 매우 weak한 모델
        clf.fit(X_train, y_train, sample_weight = w_i)

        #y_hat (각모델의 예측 y값)
        y_pred = clf.predict(X_train)

        #(b)식 알고리즘 - 전체 가중치 합중에 예측을 잘못한 가중치의 비율
        ## np.where(y_train != y_pred, w_i, 0) -> 예측값과 실제값이 같지 않다면 해당 위치값을 가중치로, 같다면 0으로 지정
        error = np.sum(np.where(y_train != y_pred, w_i, 0)) / np.sum(w_i)

        #(c)식 알고리즘 - ex)  err =0.5 => log((1-0.5)/0.5) = log1 = 0 
        ###(0.5에 가까우면 0에 가까워짐 (신뢰할 수 없기 때문에  이 모델의 a를 0에 가깝게 만든다)
        alpha =learning_rate * np.log((1-error) / error)

        #분류기 저장소에 알파와 모델을 저장한다
        classifiers.append((alpha, clf))

        #가중치 업데이트 (d)식 [핵심 알고리즘] - 못맞춘 데이터일 수록 가중치 증가
        w_i = np.where(y_train != y_pred, w_i*np.exp(alpha), w_i) 
        w_i = w_i / sum(w_i)
    return classifiers # 학습된 모델 알파와 모델 return


#Line 3 함수
def predict(clfs, x):
    s = np.zeros(len(x))
    for (alpha, clf) in clfs:
        s += alpha * clf.predict(x)
    return np.sign(s)



result = predict(AdaBoost(X_train, y_train, 3, 0.01), X_train)
print(result)


