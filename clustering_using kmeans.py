from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('dataset.csv')

f1 = data['Disease'].values
f2 = data['Symptom'].values
f3 = data['Blood Condition'].values
f4 = data['WBC'].values
f5 = data['RBC'].values

Labelencode_X = LabelEncoder()
f1 = Labelencode_X.fit_transform(f1) 

Labelencode_Y = LabelEncoder()
f2 = Labelencode_Y.fit_transform(f2)

Labelencode_Z = LabelEncoder()
f3 = Labelencode_Z.fit_transform(f3)

scaled = MinMaxScaler(copy=True, feature_range=(0, 100))
f4 = scaled.fit_transform(f4)
f5 = scaled.fit_transform(f5)

X = np.array(list(zip(f4, f5 ,f2 ,f1 ,f3 )))
plt.rcParams['figure.figsize'] = (16, 9)

# Plotting along with the Centroids
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:,2], s= 200)

# Initializing KMeans
kmeans = KMeans(n_clusters=4,init= 'random')
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:,2], c=kmeans.labels_.astype(float),s =200)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', s=200)

