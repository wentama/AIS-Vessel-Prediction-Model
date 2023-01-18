# -*- coding: utf-8 -*-
"""
Using the baseline classification algorithm given by Professor Kevin S. Xu

@Edited by: Wen Tao Lin & Mason Leung

The code for data Preprocessing, hyperparameter tuning, and other steps we took are in a separate file
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering 

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    
    scaler = StandardScaler()
    course_sin = np.sin((testFeatures[:,-1]/10)*((np.pi)/180))
    testFeatures = scaler.fit_transform(testFeatures) 
    testFeatures[:,-1] = course_sin
    
    model = SpectralClustering(n_clusters=numVessels, assign_labels="cluster_qr", n_jobs=-1, gamma=1, random_state=1)
    predVessels = model.fit_predict(testFeatures)
    
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    
    scaler = StandardScaler()
    course_sin = np.sin((testFeatures[:,-1]/10)*((np.pi)/180))
    testFeatures = scaler.fit_transform(testFeatures) 
    testFeatures[:,-1] = course_sin
    
    model = DBSCAN(eps=0.3, min_samples=8, n_jobs=-1)
    predVessels = model.fit_predict(testFeatures)
    
    return predVessels

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set3noVID.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    