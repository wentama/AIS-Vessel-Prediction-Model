# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 18:42:26 2022

@author: wtl22
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import loadData, plotVesselTracks
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from sklearn.preprocessing import StandardScaler

data = loadData('set3noVID.csv')
features = data[:,2:]
labels = data[:,1]

scaler = StandardScaler()
course_sin = np.sin((features[:,-1]/10)*((np.pi)/180))
features = scaler.fit_transform(features) 
features[:,-1] = course_sin

neigh = NearestNeighbors(n_neighbors=10)
nbrs = neigh.fit(features)
distances, indices = nbrs.kneighbors(features)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.axis([7000, 8500, 0, 0.3])
