import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid  # used for hyperParam tuning of unsurpervised model
from sklearn.cluster import KMeans, DBSCAN, MeanShift


drivers_df = pd.read_csv("./datasets/driver_details.csv")
drivers_df.head()


# best practice to check for nana values
drivers_df[drivers_df.isnull().any(axis = 1)]


drivers_df.describe()


drivers_features = drivers_df.drop("Driver_ID", axis = 1)


# tuning
parameters = {"n_clusters": [2, 3, 4, 5, 10, 20]}
parameter_grid = ParameterGrid(parameters)

# checks values in this grid obj
list(parameter_grid)


best_score = -1
model = KMeans()


# iter through paramgrid and train with hyperParams specified
for g in parameter_grid:
    model.set_params(**g)  # sets model param to hyperparam in focus
    model.fit(drivers_features)
    
    ss = metrics.silhouette_score(drivers_features, model.labels_)
    print("Parameter: ", g, "Score: ", ss)
    if ss > best_score:
        best_score = ss
        best_grid = g


best_grid


parameters = {
    "eps": [0.9, 1.0, 5.0, 10.0, 12.0, 14.0, 20.0],
    "min_samples": [5, 7, 10, 12]
}

parameter_grid = ParameterGrid(parameters)
list(parameter_grid)


model = DBSCAN()
best_score = -1


for g in parameter_grid:
    model.set_params(**g)  # sets model param to hyperparam in focus
    model.fit(drivers_features)
    
    ss = metrics.silhouette_score(drivers_features, model.labels_)
    print("Parameter: ", g, "Score: ", ss)
    if ss > best_score:
        best_score = ss
        best_grid = g


best_grid


# uses best param to now train our model proper
model.set_params(**best_grid)
model.fit(drivers_features)


len(model.labels_)   # so since in dbsca all datapoints are clusters initially


# gets the usual cluster numbers excluding the outliers with cluster label of '-1'
n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)

n_clusters


# gets noisy points or outliers
n_noise = list(model.labels_).count(-1)

n_noise


# instead of using paramGrid stuff use this for MeanShift since lin already available
from sklearn.cluster import estimate_bandwidth
estimate_bandwidth(drivers_features)


# check it
model = MeanShift(bandwidth = estimate_bandwidth(drivers_features)).fit(drivers_features)
metrics.silhouette_score(drivers_features, model.labels_)

























































