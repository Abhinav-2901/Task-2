#Task - 2 : Prediction using Unsupervised ML            
# Task 2 : Clustering using Unsupervised Learning model
# Hierarchical clustering 

# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset

dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:5].values
Y = dataset.iloc[:, 5].values

# Using the dendrogram to find the optimal number of clusters

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Features')
plt.ylabel('Ecludean Distance')
plt.show()

# Fitting the hierarchical clusrering to the mall dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage = 'ward')
hc_y = hc.fit_predict(X)

# Visualising the clusters

plt.scatter(X[hc_y == 0, 0], X[hc_y == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[hc_y == 1, 0], X[hc_y == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolor')
plt.scatter(X[hc_y == 2, 0], X[hc_y == 2, 1], s = 100, c = 'green', label = 'Iris-virginicia')
plt.title('Clustering of Iris-flower dataset')




