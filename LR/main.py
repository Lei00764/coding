"""
@File    :   main.py
@Time    :   2024/04/18 11:56:58
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   logistic Regression 算法复现 
"""

# Ref: https://realpython.com/logistic-regression-python/


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# x: independent variable
# y: dependent variable

# x = np.arange(10).reshape(-1, 1)
# 多维数据
x = np.random.randn(10, 100)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver="liblinear", random_state=0)  # 初始化一个 LR 模型

model.fit(x, y)


print(model.classes_)  # 有哪些类别/
print(model.intercept_)  # b 向量
print(model.coef_)  # w 矩阵

print(model.predict_proba(x))  # 预测结果 [[为0的概率, 为1的概率]]
print(model.predict(x))
print(model.score(x, y))
