# -*- coding: utf-8 -*-
import numpy as np
from scipy import io as spio
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances_argmin
# data = pd.read_table('./data/data_others/spiral.txt',names=['x1','x2','label'])
# data=data.drop('label',axis=1)
# print(data.head(4))
# X=data.values
# print(X)
def kMenas():
    data = pd.read_table('./data/data_others/spiral.txt', names=['x1', 'x2', 'label'])
    true_labels=data['label'].values
    data=data.drop('label',axis=1)
    X=data.values
    k_means = KMeans(n_clusters=3)# n_clusters指定3类，拟合数据
    model = k_means.fit(X)
    centroids = model.cluster_centers_  # 聚类中心

    # plt.scatter(X[:, 0], X[:, 1])  # 原数据的散点图
    # plt.plot(centroids[:, 0], centroids[:, 1], 'r^', markersize=10)  # 聚类中心
    # plt.show()

    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']


    n_clusters=3
    ax = fig.add_subplot(1,1,1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        #my_members = (true_labels-1) == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.',markersize=10)
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()



if __name__ == "__main__":
    kMenas()
