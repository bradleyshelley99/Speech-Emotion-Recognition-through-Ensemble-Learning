# Final Project L-3 Norm

This repository contains:

### L3 Norm Final Project Report in pdf

## train.py
A train.py function that will train our hybrid model taking an input data set data_training and corresponding labels labels_training

The train.py saves the hybrids models in several formats to the local directory of the user.
* CNN is saved within folder titled CNN_model
* KNN_model joblib for the model
* MLP_model joblib for the model
* Two Principal Component Analysis joblibs for the MLP and K-NN model.
* Since the model was pre-trained these files have all been included as they are needed for loading in the Juypter file 

## test.py
test.py takes in inputs data set data_testing, and desired labels labels_testing in similiar format to the training
set model.
The test.py takes the local directory joblib format files and the CNN_model folder and loads them as models and PCA transformations.
It will then use these models on an input data set data_testing, and desired labels labels_testing

## Load_Test_Function.ipynb
Load test functions is a ease of access notebook setup for use with testing the model against the project
descriptions. The juypter file only inputs needed are the path to the input data data_testing, and path to the input
labels labels_testing.

### Libraries needed
Tensorflow-2.3.1
librosa

