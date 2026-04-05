import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)


# 加载txt和csv文件
def loadtxtAndcsv_data(filename,split,dataType,skiprows=0):
    return np.loadtxt(filename,delimiter=split,dtype=dataType,skiprows=skiprows)

# 代价函数
def costFunction(theta,X,y):
    m = len(y)
    h = X @ theta
    J = np.mean((h-y)**2) / 2
    return J

# 计算梯度
def gradient(theta,X,y):
    m = len(y)
    h = X @ theta
    grad = (X.T @ (h-y)) / m
    return grad

# 梯度下降
def gradientDescent(X,y,theta,alpha,num_iters):
    J_history = []
    for i in range(num_iters):
        grad = gradient(theta,X,y)
        theta = theta - alpha * grad
        if i % 10 ==0:
            J = costFunction(theta,X,y)
            J_history.append(J)
    return theta,J_history

# 预测
def predict(X,theta):
    return X @ theta

#线性回归模型
def LinearRegression():
    print("加载数据...")
    data = loadtxtAndcsv_data("winequality-red.csv",";",np.float64,skiprows=1)
    X_raw = data[:,0:-1]
    y_raw = data[:,-1]

    m = X_raw.shape[0]
    n =X_raw.shape[1]

    print(f"样本数：{m}，特征数：{n}")

    # 归一化特征
    mu = np.mean(X_raw,axis=0)
    sigma = np.std(X_raw,axis=0)
    X_norm = (X_raw - mu) / sigma

    # 添加截距项
    X = np.hstack((np.ones((m,1)),X_norm))

    # 初始化数据
    initial_theta = np.zeros(X.shape[1])
    alpha = 0.01  # 学习率
    num_iters = 1000  # 迭代次数

    # 梯度下降
    theta,J_history = gradientDescent(X,y_raw,initial_theta,alpha,num_iters)

    # 预测
    y_pred = predict(X,theta)

    # 输出预测实例
    print("\n预测实例 前10例")
    for i in range(10):
        pred = y_pred[i]
        true = y_raw[i]
        print(f"样本{i+1}：预测={pred:.2f}，真实={true:.2f}")

    # 评估
    mse = np.mean(np.mean((y_raw - y_pred) ** 2))
    print(f"\n均方误差：{mse:.4f}")

    # 画损失曲线
    plt.figure()
    plt.plot(np.arange(0, num_iters, 10), J_history)
    plt.xlabel('迭代次数',fontproperties=font)
    plt.ylabel('损失值',fontproperties=font)
    plt.title('梯度下降损失曲线',fontproperties=font)
    plt.grid(True)
    plt.show()

    return theta


if __name__ == "__main__":
    LinearRegression()




