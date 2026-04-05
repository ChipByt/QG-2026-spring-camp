import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

# 加载txt和csv文件
def loadtxtAndcsv_data(filename,split,dataType,skiprows=0):
    return np.loadtxt(filename,delimiter=split,dtype=dataType,skiprows=skiprows)

# S型函数
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# 代价函数
def costFunction(theta,X,y,lambda_reg):
    m =len(y)
    h = sigmoid(X @ theta)
    theta_reg = theta.copy()
    theta_reg[0] = 0
    J = (-y * np.log(h + 1e-8) - (1-y)*np.log(1-h+1e-8)).mean()
    J += (lambda_reg/(2*m)) * np.sum(theta_reg ** 2)
    return J

# 计算梯度
def gradient(theta,X,y,lambda_reg):
    m = len(y)
    h = sigmoid(X @ theta)
    theta_reg = theta.copy()
    theta_reg[0] = 0
    grad = (X.T @ (h-y)) / m
    grad +=(lambda_reg/m) * theta_reg
    return grad

# 梯度下降
def gradientDescent(X,y,theta,alpha,num_iters,lambda_reg):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        grad = gradient(theta,X,y,lambda_reg)
        theta = theta - alpha * grad
        if  i % 10 == 0:
            J = costFunction(theta,X,y,lambda_reg)
            J_history.append(J)
    return theta,J_history

# 预测
def predict(X,theta):
    return (sigmoid(X @ theta)>0.5).astype(int)

# 模型评估
def evaluation_metrics(y_true,y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # 准确率
    accuracy = ((tp + tn) / len(y_true)) * 100

    # 查准率
    precision = tp / (tp + fp)

    # 查全率
    recall = tp / (tp + fn)

    # F1
    f1 = 2 * precision * recall / (precision + recall)

    #混淆矩阵
    confusion = np.array([[tn, fp], [fn, tp]])

    return accuracy,precision,recall,f1,confusion


# 逻辑回归模型
def LogisticRegression():
    print("加载数据...")
    # 加载红酒数据
    data = loadtxtAndcsv_data("winequality-red.csv",";",np.float64,skiprows=1)
    X_raw = data[:,0:-1]
    y_raw = data[:,-1]

    # 变为二分类
    y = (y_raw > 6).astype(int)

    m = X_raw.shape[0]
    n = X_raw.shape[1]

    print(f"样本数：{m}，特征数：{n}")
    print(f"好酒数量：{np.sum(y)}，坏酒数量：{m - np.sum(y)}")

    # 归一化特征
    mu = np.mean(X_raw,axis=0)
    sigma = np.std(X_raw,axis=0)
    X_norm = (X_raw - mu) / sigma

    # 添加截距项
    X = np.hstack((np.ones((m,1)),X_norm))

    # 初始化参数
    initial_theta = np.zeros(X.shape[1])
    alpha = 0.1  # 学习率
    num_iters = 1000  # 迭代次数
    lambda_reg = 0.1  # 正则化系数

    print("\n开始梯度下降...")
    theta,J_history = gradientDescent(X,y,initial_theta,alpha,num_iters,lambda_reg)

    # 预测
    y_pred = predict(X,theta)

    # 输出预测实例
    print("\n预测实例（前10个样本）：")
    for i in range(10):
        prob = sigmoid(X[i] @ theta)
        pred = "好酒" if prob >= 0.5 else "坏酒"
        true = "好酒" if y[i] == 1 else "坏酒"
        print(f"样本{i+1}：预测={pred}({prob:.4f},真实={true})")

    # 模型评估
    print("\n模型评估结果：")
    accuracy, precision, recall, f1, confusion = evaluation_metrics(y, y_pred)
    print(f"准确率: {accuracy:.2f}%")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    print("\n混淆矩阵:")
    print("           预测坏酒  预测好酒")
    print(f"实际坏酒:     {confusion[0, 0]}        {confusion[0, 1]}")
    print(f"实际好酒:     {confusion[1, 0]}        {confusion[1, 1]}")

    # 画损失曲线
    plt.figure()
    plt.plot(np.arange(0, num_iters, 10), J_history)
    plt.xlabel('迭代次数', fontproperties=font)
    plt.ylabel('损失值', fontproperties=font)
    plt.title('梯度下降损失曲线', fontproperties=font)
    plt.grid(True)
    plt.show()

    return theta


if __name__ == "__main__":
    LogisticRegression()

