# -*- coding: utf-8 -*-
"""
Script to evaluate prediction accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from utils import loadData, plotVesselTracks
from predictVessel import predictWithK, predictWithoutK
from sklearn.preprocessing import StandardScaler

#%% Load training and test data. Training data may not necessarily be used.
testData = loadData('set2.csv')
testFeatures = testData[:,2:]
testLabels = testData[:,1]
trainData = np.r_[loadData('set1.csv'), loadData('set2.csv')]
trainFeatures = trainData[:,2:]
trainLabels = trainData[:,1]

#%% Run prediction algorithms and check accuracy
numVessels = np.unique(testLabels).size
predVesselsWithK = predictWithK(testFeatures, numVessels, trainFeatures, 
                                trainLabels)
# Check to ensure that there are at most K vessels. If not, set adjusted
# Rand index to -infinity to indicate an invalid result (0 accuracy score)
if np.unique(predVesselsWithK).size > numVessels:
    ariWithK = -np.inf
else:
    ariWithK = adjusted_rand_score(testLabels, predVesselsWithK)

predVesselsWithoutK = predictWithoutK(testFeatures, trainFeatures, trainLabels)
predNumVessels = np.unique(predVesselsWithoutK).size
ariWithoutK = adjusted_rand_score(testLabels, predVesselsWithoutK)

print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
      + f'{ariWithoutK}')

#%% Plot vessel tracks colored by prediction and actual labels


plt.ion()
plotVesselTracks(testFeatures[:,[2,1]], predVesselsWithK)
plt.title('Vessel tracks by cluster with K')
plotVesselTracks(testFeatures[:,[2,1]], predVesselsWithoutK)
plt.title('Vessel tracks by cluster without K')
plotVesselTracks(testFeatures[:,[2,1]], testLabels)
plt.title('Vessel tracks by label')
