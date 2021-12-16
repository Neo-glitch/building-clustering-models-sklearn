import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics


mnist_data = pd.read_csv("./datasets/train.csv")
mnist_data.head()


mnist_features = mnist_data.drop("label", axis = 1)
mnist_labels = mnist_data.label

mnist_features.head()


# helper fn to display image and the label
def display_image(index):
    print("Digit is : ", mnist_labels[index])
    plt.imshow(mnist_features.loc[index].values.reshape(28, 28), cmap = "Greys")


display_image(10)


# main meat
kmeans_model = KMeans(n_clusters = 10, max_iter = 1000).fit(mnist_features)


kmeans_centroids = kmeans_model.cluster_centers_

kmeans_centroids


# plot images that make up the cluster centers
fig, ax = plt.subplots(figsize = (10, 10))

for centroid in range(len(kmeans_centroids)):
    plt.subplot(2, 5, centroid + 1)
    
    plt.imshow(kmeans_centroids[centroid].reshape(28, 28), cmap = "Greys")


mnist_test = mnist_data.sample(10, replace = False)
mnist_test_features = mnist_test.drop("label", axis = 1)
mnist_test_labels = mnist_test["label"]


mnist_test_labels


# conv labels datafram to np
mnist_test_labels = np.array(mnist_test_labels)
mnist_test_labels


# prediction
pred_clusters = kmeans_model.predict(mnist_test_features)


pred_results = pd.DataFrame({
    "actual_digit": mnist_test_labels,
    "pred_cluster": pred_clusters
})


# cross check using cluster center image above
pred_results.head(10)































































