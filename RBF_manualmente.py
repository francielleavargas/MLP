#/usr/bin/python3.7

import numpy as np
import pandas as pd 
import random
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#loading dataset
dataset = pd.read_csv('iris.csv')
#print(dataset.head())

#separating features and class
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:,-1]

#preparating features for newtwork
data = np.array( X, dtype=np.float32 )
labels = np.array( Y, dtype=np.int32 )

#preparating labels for newtwork
num_categories = 3
new_labels = np.zeros( [ len(labels), num_categories ] )
for i in range( len(labels) ):
  new_labels[ i, labels[i]-1 ] = 1.
labels = new_labels


#print features and labels
print('---features---')
print(data)
print('---class---')
print(labels)


#separating test and training data
validation_size = 0.30
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(data, labels, test_size=validation_size, random_state=seed, shuffle= True)


#normalization
scaler = MinMaxScaler( (-1,1) )
X_train = scaler.fit_transform( X_train )
X_validation = scaler.transform( X_validation )
print(X_train.shape, X_validation.shape)


#RBF NETWORK
from sklearn.cluster import KMeans

###initialize the weights of the intermediate layer
def initialize_centroids( n_centroids, X ):
    kmeans = KMeans( n_clusters = n_centroids ).fit(X)
    centroids = kmeans.cluster_centers_
    return centroids

###initialize the weights of the output layer
def initialize_weights( n_centroids, n_outputs ):
    W = np.random.normal( loc=0, scale=0.1, size=( n_centroids, n_outputs ) )
    return W

###sigmoid  function
def gaussian( C, X, sigma=1. ):
    dists = np.sqrt( np.sum( (X-C)**2, axis=1 ) )
    return np.exp(-dists**2 / (2 * sigma**2))

###step function
def step( v ):
    if v > 0:
        return 1
    return 0

###forward
def forward( C, W, X ):
    phi = gaussian( C, X )
    V = np.dot( phi, W )
    Y = [step(v) for v in V]
    Y = np.array( Y )
    return Y

###prediction
def predict( C, W, data ):
    outputs = list()
    for X in data:
        Y = forward( C, W, X )
        outputs.append( Y )
    return outputs

##acurracy
def evaluate( C, W, data, t ):
    Y = predict( C, W, data )
    hits = np.sum( [ np.argmax(Y[i]) == np.argmax(t[i]) for i in range( len(Y) ) ] )
    acc = hits / len(Y)
    return acc

###root-mean-square deviation (RMSD) for each simple
def compute_mse( y, t ):
  return 1/2 * np.sum( [ (t[i] - y[i])**2 for i in range(len(y)) ] )

###root-mean-square deviation (RMSD) for dataset
def compute_total_mse( C, W, data, labels ):
  y = predict( C, W, data )
  E = [ compute_mse( y[i], labels[i] ) for i in range(len(data)) ]
  return np.mean( E )

###trainning
def train( X_train, Y_train, n_centroids, sigma=1.2, eta=0.1, epochs=1000, epsilon=0.1 ):
  # Camada intermediária
  C = initialize_centroids( n_centroids, X_train )
  # Camada de saída
  W = initialize_weights( n_centroids, Y_train.shape[1] )
  nRows = X_train.shape[0]
  error = np.inf
  for epoch in range( epochs ):
    if error < epsilon:
      break
    new_W = W
    for i in range( nRows ):
      Y = forward( C, W, X_train[i] )
      for j in range( Y_train.shape[1] ):
        new_W[:,j] += eta * gaussian( X_train[i], C, sigma ) * ( Y_train[i,j]-Y[j] )
    W = new_W
    error = compute_total_mse( C, W, X_train, Y_train )
    if not epoch % 200:
      print(epoch, error)
  return C, W


########RESULTS
C, W = train( X_train, Y_train, n_centroids=2, sigma=1.0, epochs=1000, epsilon=0.1 )
print( 'Train accuracy:', evaluate( C, W, X_train, Y_train ) )
print( 'Test accuracy:', evaluate( C, W, X_validation, Y_validation ) )