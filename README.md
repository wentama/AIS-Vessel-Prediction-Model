# AIS-Vessel-Prediction-Model
Two machine learning algorithms to track various moving vessels using Automatic Identification System (AIS) data including a vessel’s latitude, longitude, speed over ground, course over ground labeled with time stamps. One model, DBSCAN, is used when we are not given the number of vessels in the dataset. And the other model, Spectral Clustering, is used when we are given the number of vessels.

# Evaluation 

Round 1 Evaluation: set1 is the training data. And set2noVID is also provided. 
                    set2 is the testing data.

Final Evaluation: set1 + set2 is the training data. set2noVID + set3noVID provided. 
                  set3 is the testing data.

*Note* since we chose unsupervising learning. Training data was not used to train the model.
