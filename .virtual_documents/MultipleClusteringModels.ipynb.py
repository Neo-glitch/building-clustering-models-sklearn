import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, \
    Birch, AffinityPropagation, MiniBatchKMeans
import warnings
warnings.filterwarnings("ignore")


iris_df = pd.read_csv("./datasets/iris.csv", skiprows = 1,
                     names = ["sepal-length", "sepal-width",
                             "petal-length", "petal-width", "class"])

iris_df = iris_df.sample(frac=1).reset_index(drop=True)
iris_df.head()


# cat to num
from sklearn import preprocessing

label_encoding = preprocessing.LabelEncoder()
iris_df["class"] = label_encoding.fit_transform(iris_df["class"].astype(str))

iris_df.head()


iris_features = iris_df.drop("class", axis = 1)

iris_features.head()


iris_labels = iris_df["class"]

iris_labels.sample(5)


# helper fun to train and eval clust model score
def build_model(clustering_model, data, labels):
    model = clustering_model(data)
    
    print("homo\tcompl\tv-score\tARI\tANI\tsilhouette")
    print(50 * "-")
    print('get_ipython().run_line_magic(".3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'", "")
        get_ipython().run_line_magic("(metrics.homogeneity_score(labels,", " model.labels_),")
          metrics.completeness_score(labels, model.labels_),
          metrics.v_measure_score(labels, model.labels_),
          metrics.adjusted_rand_score(labels, model.labels_),
          metrics.adjusted_mutual_info_score(labels, model.labels_),
          metrics.silhouette_score(data, model.labels_)))


def k_means(data, n_clusters = 3, max_iter = 1000):
    model = KMeans(n_clusters = n_clusters, max_iter = max_iter).fit(data)
    return model
    


# use the helper fun
build_model(k_means, iris_features, iris_labels)


def agglomerative_fn(data, n_clusters = 3):
    model = AgglomerativeClustering(n_clusters=n_clusters).fit(data)
    return model

build_model(agglomerative_fn, iris_features, iris_labels)


def dbscan_fn(data, eps = 0.45, min_samples = 4):
    # eps = 0.45 means all points within this distance are neighbors and can be in same cluster
    # min samples = 4, means for region to be considered a cluster it must have 4 close data points or more
    model = DBSCAN(eps = eps, min_samples=min_samples).fit(data)
    return model

build_model(dbscan_fn, iris_features, iris_labels)


def mean_shift_fn(data, bandwidth = 0.85):
    model = MeanShift(bandwidth=bandwidth).fit(data)
    return model

build_model(mean_shift_fn, iris_features, iris_labels)


def birch_fn(data, n_clusters = 3):
    model = Birch(n_clusters = n_clusters).fit(data)
    return model

build_model(birch_fn, iris_features, iris_labels)


def affinity_propagation_fn(data, damping=0.6, max_iter = 1000):
    # damping is like the lr, that's whether a data point remains in current exampler or update to new one
    model = AffinityPropagation(damping = damping, max_iter = max_iter).fit(data)
    return model

build_model(affinity_propagation_fn, iris_features, iris_labels)


def mini_batch_kmeans_fn(data, n_clusters = 3):
    # faster than normal kmeans, but slightly less acc than full Kmeans
    model = MiniBatchKMeans(n_clusters=n_clusters).fit(data)
    return model

build_model(mini_batch_kmeans_fn, iris_features, iris_labels)


# spectral clustering can accept a precomputed affinity matrix or raw data as before
from sklearn.cluster import SpectralClustering


SS = 1000 # self-similarity
IS= 10 # intra-cluster similarity
LS = 0.01 # similarity btw points in diff clusters


# each row corresponds to every point in dataset
# each col corresponds to every point in dataset
similarity_mat = [
    [SS, IS, IS, LS, LS, LS],
    [IS, SS, IS, LS, LS, LS],
    [IS, IS, SS, LS, LS, LS],
    [LS, LS, LS, SS, IS, IS],
    [LS, LS, LS, IS, SS, LS],
    [LS, LS, LS, LS, LS, SS]
]

# affinity = precomputed since we are passing the affinity mat, if raw data use default
spectral_model = SpectralClustering(n_clusters = 3, affinity="precomputed").fit(similarity_mat)
spectral_model.labels_
























































































































