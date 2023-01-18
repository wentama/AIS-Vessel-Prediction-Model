# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 01:02:30 2022

@author: wtl22
"""

import numpy as np
import scipy.cluster.hierarchy as hc
from utils import loadData, plotVesselTracks
import matplotlib.pyplot as plt

data = loadData('set1.csv')
features = data[:,2:]
labels = data[:,1]

single_link = hc.linkage(data, method='single', metric='euclidean')
plt.figure()
hc.dendrogram(single_link, truncate_mode=None)


model = GaussianMixture(n_components=numVessels, random_state=1)

model = MiniBatchKMeans(n_clusters=numVessels, random_state=1, batch_size=2048)

model = KMeans(n_clusters=numVessels, random_state=100)

model = Birch (n_clusters=numVessels)

model = DBSCAN()

model = SpectralClustering(n_jobs=-1, n_clusters=numVessels, n_neighbors=5, assign_labels="cluster_qr" )

scaler = StandardScaler()
course_sin =np.sin((testFeatures[:,-1]/10)*((np.pi)/180))
print(course_sin)
print("------------------------")
print(testFeatures)
print("------------------------")
testFeatures = scaler.fit_transform(testFeatures)
print(testFeatures)
print("------------------------")
testFeatures[:,-1] = course_sin
print(testFeatures)
print("------------------------")
model = SpectralClustering(n_jobs=-1, n_clusters=numVessels, n_neighbors=5, assign_labels="cluster_qr" )
predVessels = model.fit_predict(testFeatures)

scaler = StandardScaler()
course_sin = np.sin((testFeatures[:,-1]/10)*((np.pi)/180))
testFeatures = scaler.fit_transform(testFeatures) 
testFeatures[:,-1] = course_sin
model = DBSCAN(eps = 0.1, n_jobs=-1)

predVessels = model.fit_predict(testFeatures)


course_sin = np.sin((features[:,-1]/10)*((np.pi)/180))
features = scaler.fit_transform(features) 
features[:,-1] = course_sin

plt.axis([2500, 3500, 0, 0.25])

plt.axis([12000, 14000, 0, 0.25])

scaler = StandardScaler()
course_sin = np.sin((testFeatures[:,-1]/10)*((np.pi)/180))
testFeatures = scaler.fit_transform(testFeatures) 
testFeatures[:,-1] = course_sin
model = SpectralClustering(n_jobs=-1, n_clusters=numVessels, n_neighbors=5, assign_labels="cluster_qr" )
predVessels = model.fit_predict(testFeatures)


testFeatures[:,5] = np.sum(testFeatures[:, [1,2,3]], axis=1])


scaler = StandardScaler()
course_sin = np.sin((testFeatures[:,4]/10)*((np.pi)/180))
sum_features = testFeatures[:,1] + testFeatures[:,2] +testFeatures[:,3]
print(sum_features)
print("________________________")
testFeatures = np.append(testFeatures, np.sum(testFeatures[:, [1,2,3]], axis = 1).reshape(np.shape(testFeatures)[0],-1), axis = 1)
print(testFeatures)
print("________________________")
testFeatures = scaler.fit_transform(testFeatures) 
testFeatures[:,4] = course_sin
print(testFeatures)
print("________________________")
print(testFeatures[:,[4,5]])


model = SpectralClustering(n_jobs=-1, n_clusters=numVessels, n_neighbors=8, assign_labels="kmeans" )
predVessels = model.fit_predict(testFeatures[:,[4,5]])

plt.scatter(testFeatures[:,4], testFeatures[:,5], c=predVessels)

scaler = StandardScaler()
course_sin = np.sin((testFeatures[:,4]/10)*((np.pi)/180))
sum_features = testFeatures[:,1] + testFeatures[:,2] +testFeatures[:,3]
print(sum_features)
print("________________________")
testFeatures = np.append(testFeatures, np.sum(testFeatures[:, [1,2,3]], axis = 1).reshape(np.shape(testFeatures)[0],-1), axis = 1)
print(testFeatures)
print("________________________")
testFeatures = scaler.fit_transform(testFeatures) 
testFeatures[:,4] = course_sin