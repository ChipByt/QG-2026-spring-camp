import numpy as np
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)


# 加载文件数据
def load_iris_data(filename,split,dataType):
    return np.loadtxt(filename,delimiter=split,dtype=dataType,usecols=range(4))

# 加载真实标签
def load_iris_labels(filename):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) == 5:
                    if parts[4] == 'Iris-setosa':
                        labels.append(0)
                    elif parts[4] == 'Iris-versicolor':
                        labels.append(1)
                    elif parts[4] == 'Iris-virginica':
                        labels.append(2)
    return np.array(labels)

# 随机初始化K个中心
def init_centroids(X,K):
    np.random.seed(42)
    indices = np.random.choice(len(X),K,replace=False)
    return X[indices]

# 分配样本到最近的中心
def find_closest_centroid(X,centroids):
    m = X.shape[0]
    K = centroids.shape[0]
    labels = np.zeros(m,dtype=int)
    for i in range(m):
        # 计算到所有中心的距离
        distances = np.sqrt(np.sum((X[i]-centroids)**2,axis=1))
        labels[i] = np.argmin(distances)
    return labels

# 更新中心
def compute_centroids(X,labels,K):
    n = X.shape[1]
    centroids = np.zeros((K,n))
    for k in range(K):
        points = X[labels==k]
        if len(points)>0:
            centroids[k] = np.mean(points,axis=0)
        else:
            centroids[k] = X[np.random.choice(len(X))]  # 如果某个簇是空的，随机重新初始化
    return centroids

# K-Means算法
def KMeans(X,K,max_iters=100):
    centroids = init_centroids(X,K)
    for i in range(max_iters):
        old_centroids = centroids.copy()
        labels = find_closest_centroid(X,centroids)
        centroids = compute_centroids(X,labels,K)

        # 检查是否收敛
        if np.all(old_centroids == centroids):
            print(f"收敛于第{i+1}次迭代")
            break
    return centroids,labels

# 评估函数：rand指数
def rand(true_labels,pred_labels):
    n = len(true_labels)
    a = b = c = d = 0
    for i in range(n):
        for j in range(i+1,n):
            same_pred = (pred_labels[i] == pred_labels[j])
            same_true = (true_labels[i] == true_labels[j])
            if same_pred and same_true:
                a += 1
            elif same_pred and not same_true:
                b += 1
            elif not same_pred and same_true:
                c += 1
            else:
                d += 1
    return (a+d) / (a+b+c+d)



if __name__=='__main__':
    X = load_iris_data('iris.data.txt',",",np.float64)
    true_labels = load_iris_labels('iris.data.txt')
    print(f"数据形状：{X.shape}")

    K = 3
    max_iters = 100

    print("正在运行K_Means...")
    centroids,labels = KMeans(X,K,max_iters)

    # 评估
    ri = rand(true_labels, labels)
    print(f"\nRand指数: {ri:.4f}")
    print(f"\n每个簇的样本数: {np.bincount(labels)}")
    print(f"\n簇中心坐标:")
    for k in range(K):
        print(f"  簇 {k + 1}: {centroids[k]}")

    # 可视化结果（两个子图）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['red', 'blue', 'green']
    # 图1：花萼信息
    for k in range(K):
        ax1.scatter(X[labels == k, 0], X[labels == k, 1],
                    c=colors[k], label=f'cluster {k + 1}', alpha=0.6)
    ax1.scatter(centroids[:, 0], centroids[:, 1],
                c='black', marker='x', s=200, linewidths=3, label='centroid')
    ax1.set_xlabel('花萼长度', fontproperties=font)
    ax1.set_ylabel('花萼宽度', fontproperties=font)
    ax1.set_title('聚类结果（花萼）', fontproperties=font)
    ax1.legend()
    ax1.grid(True)
    # 图2：花瓣信息
    for k in range(K):
        ax2.scatter(X[labels == k, 2], X[labels == k, 3],
                    c=colors[k], label=f'簇 {k + 1}', alpha=0.6)
    ax2.scatter(centroids[:, 2], centroids[:, 3],
                c='black', marker='x', s=200, linewidths=3, label='簇中心')
    ax2.set_xlabel('花瓣长度', fontproperties=font)
    ax2.set_ylabel('花瓣宽度', fontproperties=font)
    ax2.set_title('聚类结果（花瓣）', fontproperties=font)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


