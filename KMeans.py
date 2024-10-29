import random
import math


# 计算两个点之间的欧氏距离
def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))


# 计算一个簇的质心
def compute_centroid(cluster):
    num_points = len(cluster)
    centroid = [sum(dimension) / num_points for dimension in zip(*cluster)]
    return centroid


# 将每个点分配给最近的质心
def assign_points_to_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid_index = distances.index(min(distances))
        clusters[closest_centroid_index].append(point)
    return clusters


# K-Means算法
def kmeans(data, k, max_iterations=100):
    # 随机初始化质心
    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        # 分配数据点到最近的质心
        clusters = assign_points_to_clusters(data, centroids)

        # 计算新的质心
        new_centroids = [compute_centroid(cluster) for cluster in clusters]

        # 如果质心不再变化，终止迭代
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return clusters, centroids


# 测试数据
data = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0]
]

# 运行K-Means算法，指定k为3
clusters, centroids = kmeans(data, k=3)

print("聚类结果：")
for i, cluster in enumerate(clusters):
    print(f"簇 {i + 1}: {cluster}")

print("\n质心：")
for i, centroid in enumerate(centroids):
    print(f"质心 {i + 1}: {centroid}")

print("aa")