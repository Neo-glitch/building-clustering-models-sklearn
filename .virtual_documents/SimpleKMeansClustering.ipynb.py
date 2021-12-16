from sklearn import metrics
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# to avoid warnings
import warnings
warnings.filterwarnings("ignore")


# generates dataset of random numbers

# gen 50 numbers with X and Y cords, first range gen X and Y cords
data_1 = np.array([[np.random.randint(1, 400) for i in range(2)] for j in range(50)], dtype = np.float64)

data_2 = np.array([[np.random.randint(300, 700) for i in range(2)] for j in range(50)], dtype = np.float64)

data_3 = np.array([[np.random.randint(600, 900) for i in range(2)] for j in range(50)], dtype = np.float64)

data = np.append(np.append(data_1, data_2, axis = 0), data_3, axis = 0)


data.shape


# viz data
fig, ax = plt.subplots(figsize = (8, 8))
plt.scatter(data[:, 0], data[:, 1], s = 100)


# labels for our datapoints
labels_1 = np.array([0 for i in range(50)])  # for data_1

labels_2 = np.array([1 for i in range(50)])  # for data_2

labels_3 = np.array([2 for i in range(50)])  # for data_3

labels = np.append(np.append(labels_1, labels_2, axis = 0), labels_3, axis = 0)


labels


# creates dataFrame since easier to work with that, for data wranggline or viz
df = pd.DataFrame({"data_x": data[:,0], "data_y": data[:, 1], "labels": labels})

df.sample(10)


# viz data points but with color based on labels data falls under
colors = ["green", "blue", "purple"]

plt.figure(figsize=(10, 10))

plt.scatter(df["data_x"], df["data_y"], c= df["labels"], s = 100,
           cmap = matplotlib.colors.ListedColormap(colors))


# kmeans model
kmeans_model = KMeans(n_clusters=3, max_iter = 10000).fit(data)


kmeans_model.labels_


# gets the cluster centroids of the 3 clusters created by model
centroids = kmeans_model.cluster_centers_

centroids


# viz where cluster centers are
fig, ax = plt.subplots(figsize = (10, 10))

plt.scatter(centroids[:,0], centroids[:,1], c='r', s= 100, marker="*")

for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] + 7, centroids[i][1] + 7), fontsize = 15)


print(f"Homogenity score: {metrics.homogeneity_score(labels, kmeans_model.labels_)}")

print(f"Completeness score: {metrics.completeness_score(labels, kmeans_model.labels_)}")

print(f"V_measure score: {metrics.v_measure_score(labels, kmeans_model.labels_)}")

print(f"Adjusted Rand Score: {metrics.adjusted_rand_score(labels, kmeans_model.labels_)}")

print(f"Adjusted_mutual_info score: {metrics.adjusted_mutual_info_score(labels, kmeans_model.labels_)}")

# doesn't req actual label
print(f"Silhouette score: {metrics.silhouette_score(data, kmeans_model.labels_)}")


# viz how this clustering turn out
colors = ["green", "blue", "purple"]

plt.figure(figsize=(10, 10))

plt.scatter(df["data_x"], df["data_y"], c= df["labels"], s = 100,
           cmap = matplotlib.colors.ListedColormap(colors), alpha = 0.6)


# does work to identify centroids and annot
plt.scatter(centroids[:,0], centroids[:,1], c='r', s= 100, marker="*")
for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] + 7, centroids[i][1] + 7), fontsize = 15)


# prediciton (to predict cluster point belongs to)
data_test = np.array([
    [442., 621.],[50., 153.],
    [333., 373.], [835., 816.]
])

label_pred = kmeans_model.predict(data_test)

label_pred


# plot test data point to see cluster it belongs to
colors = ["green", "blue", "purple"]

plt.figure(figsize=(10, 10))

plt.scatter(df["data_x"], df["data_y"], c= df["labels"], s = 100,
           cmap = matplotlib.colors.ListedColormap(colors), alpha = 0.6)

# plots test dataPoint
plt.scatter(data_test[:,0], data_test[:,1], c="orange", s= 100, marker="o")
for i in range(len(label_pred)):
    plt.annotate(label_pred[i], (data_test[i][0] + 7, data_test[i][1] - 7), fontsize =15)


# does work to identify centroids and annot
plt.scatter(centroids[:,0], centroids[:,1], c='r', s= 100, marker="*")
for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] + 7, centroids[i][1] + 7), fontsize = 15)


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


iris_df = pd.read_csv("./datasets/iris.csv", skiprows = 1,
                     names = ["sepal-length", "sepal-width",
                             "petal-length", "petal-width", "class"])

iris_df.head()


# shuffles dataframe
iris_df = iris_df.sample(frac=1).reset_index(drop = True)
iris_df.head()


# gets class cat
iris_df["class"].unique()


# encode cat
from sklearn import preprocessing

label_encoding = preprocessing.LabelEncoder()
iris_df["class"] = label_encoding.fit_transform(iris_df["class"].astype(str))
iris_df.head()


label_encoding.classes_


# viz dataset
fig, ax = plt.subplots(figsize = (8, 8))
plt.scatter(iris_df["sepal-length"], iris_df["sepal-width"], s= 100)

plt.xlabel("sepal-length")
plt.ylabel("sepal-wdith")
plt.show();


# viz dataset
fig, ax = plt.subplots(figsize = (8, 8))
plt.scatter(iris_df["petal-length"], iris_df["petal-width"], s= 100)

plt.xlabel("petal-length")
plt.ylabel("petal-width")
plt.show();


# viz dataset
fig, ax = plt.subplots(figsize = (8, 8))
plt.scatter(iris_df["sepal-length"], iris_df["petal-length"], s= 100)

plt.xlabel("sepal-length")
plt.ylabel("petal-length")
plt.show();


iris_2D = iris_df[["sepal-length", "petal-length"]]

iris_2D.sample(5)


# conv to np array to feed to model
iris_2D = np.array(iris_2D)

kmeans_model_2D = KMeans(n_clusters = 3, max_iter=1000).fit(iris_2D)


kmeans_model_2D.labels_


# plot cluster centers
centroids_2D = kmeans_model_2D.cluster_centers_

# does work to identify centroids and annot
plt.scatter(centroids_2D[:,0], centroids_2D[:,1], c='r', s= 100, marker="*")
for i in range(len(centroids_2D)):
    plt.annotate(i, (centroids_2D[i][0], centroids_2D[i][1]), fontsize = 15)


# eval clustering
iris_labels = iris_df["class"]

print(f"Homogenity score: {metrics.homogeneity_score(iris_labels, kmeans_model_2D.labels_)}")

print(f"Completeness score: {metrics.completeness_score(iris_labels, kmeans_model_2D.labels_)}")

print(f"V_measure score: {metrics.v_measure_score(iris_labels, kmeans_model_2D.labels_)}")

print(f"Adjusted Rand Score: {metrics.adjusted_rand_score(iris_labels, kmeans_model_2D.labels_)}")

print(f"Adjusted_mutual_info score: {metrics.adjusted_mutual_info_score(iris_labels, kmeans_model_2D.labels_)}")

# doesn't req actual label
print(f"Silhouette score: {metrics.silhouette_score(iris_2D, kmeans_model_2D.labels_)}")


# viz how this clustering turn out
colors = ["yellow", "blue", "green"]

plt.figure(figsize=(10, 10))

plt.scatter(iris_df["sepal-length"], iris_df["petal-length"], c= iris_df["class"], s = 100,
           cmap = matplotlib.colors.ListedColormap(colors), alpha = 0.6)


# does work to identify centroids and annot
plt.scatter(centroids_2D[:,0], centroids_2D[:,1], c='r', s= 100, marker="*")
for i in range(len(centroids_2D)):
    plt.annotate(i, (centroids_2D[i][0], centroids_2D[i][1]), fontsize = 15)


iris_features = iris_df.drop("class", axis = 1)
iris_features.head()


# get target
iris_labels = iris_df["class"]


kmeans_model = KMeans(n_clusters = 3).fit(iris_features)


# plot cluster centers
centroids = kmeans_model.cluster_centers_

# does work to identify centroids and annot
plt.scatter(centroids[:,0], centroids[:,1], c='r', s= 100, marker="*")
for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0], centroids[i][1]), fontsize = 15)


# eval clustering
print(f"Homogenity score: {metrics.homogeneity_score(iris_labels, kmeans_model.labels_)}")

print(f"Completeness score: {metrics.completeness_score(iris_labels, kmeans_model.labels_)}")

print(f"V_measure score: {metrics.v_measure_score(iris_labels, kmeans_model.labels_)}")

print(f"Adjusted Rand Score: {metrics.adjusted_rand_score(iris_labels, kmeans_model.labels_)}")

print(f"Adjusted_mutual_info score: {metrics.adjusted_mutual_info_score(iris_labels, kmeans_model.labels_)}")

# doesn't req actual label
print(f"Silhouette score: {metrics.silhouette_score(iris_features, kmeans_model.labels_)}")










































