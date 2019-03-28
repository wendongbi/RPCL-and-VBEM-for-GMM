#!/usr/bin/python
# coding=utf-8
from numpy import *
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
# generate data
def makeDataSet(k=3):
    datMat = ones([90, 2])
    x_mean = [1, 5, 10]
    y_mean = [1, 5, 1]
    x_value =[]
    y_value = []
    for meanX in x_mean:
        x = random.normal(meanX, 1, [30,1])
        x_value.extend(x)
    for meanY in y_mean:
        y = random.normal(meanY, 1, [30,1])
        y_value.extend(y)
    for i in range(90):
        datMat[i, 0] = x_value[i]
        datMat[i, 1] = y_value[i]

    datMat = array(datMat)
    savetxt('data.txt', datMat)
    print('save sucess')

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 求两个向量之间的距离


# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1] # (90, 2)

    centroids = mat(zeros((k, n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


# k-means 聚类算法
def RPCL(dataSet, k, dist_func=distEclud, createCent=randCent, lr=1, penalty=0.5):
    m = shape(dataSet)[0]
    dist_to_mu = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
    # dist_to_mu第一列存放该数据所属的cluster index，第二列是该数据到中心点的距离, 第三列是该数据第二接近的cluster index
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛,false为收敛
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf  # 最小
            minIndex = -1
            for j in range(k):
                distJI = dist_func(centroids[j, :], dataSet[i, :])  # 算第i个数据点到第k个聚类中心的距离
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if dist_to_mu[i, 0] != minIndex : clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            dist_to_mu[i, :] = minIndex, minDist**2  # 并将第i个数据点的分配情况存入字典
        # print(centroids)
        for cent in range(k):   # 重新计算中心点
            # penalize the reval
            secDist = inf  # 第二小
            secIndex = -1
            for i in range(k):
                if i != cent:
                    dist = dist_func(centroids[cent, :], centroids[i, :])
                    if dist < secDist:
                        secDist = dist
                        secIndex = i


            ptsInClust = dataSet[nonzero(dist_to_mu[:, 0] == cent)[0]]   # points in cluster
            # for i in range(m):
            #     if(dataSet[i, 0] == cent):
            
            centroids[cent, :] = centroids[cent, :] + lr * (mean(ptsInClust, axis=0) - centroids[cent, :])  # 算出这些数据的中心点
            centroids[secIndex, :] = centroids[secIndex, :] - lr * penalty * (mean(ptsInClust, axis=0) - centroids[secIndex, :])


    return centroids, dist_to_mu
# --------------------测试----------------------------------------------------
# 用测试数据及测试RPCL算法
# makeDataSet()
datMat = genfromtxt(fname='data.txt',
                            dtype=float)

myCentroids, dist_to_mu = RPCL(datMat, 5)
print(myCentroids)
print(shape(myCentroids))
# print(dist_to_mu)
print(shape(dist_to_mu))


mu_x = [0, 0, 0, 0, 0]
mu_y = [0, 0, 0, 0, 0]
color = ['b', 'y', 'g']
for i in range(5):
    mu_x[i] = myCentroids[i, 0]
    mu_y[i] = myCentroids[i, 1]
for i in range(3):
    plt.scatter(datMat[i*30:i*30+30, 0], datMat[i*30:i*30+30:, 1], c=color[i], alpha=0.6)

plt.scatter(mu_x, mu_y, c='r', marker='>', s=50)
plt.show()
