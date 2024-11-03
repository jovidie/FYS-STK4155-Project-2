
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
pd.set_option('future.no_silent_downcasting', True)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, ReLU, softmax
from ptwo.costfuns import binary_cross_entropy


def main():
    #importing data:
    data = pd.read_csv("./data/wisconsin_breast_cancer_data.csv")
    print(f"DATA\n{data}")

    # preprocessing data: 
    targets = data["diagnosis"]
    network_input = data.iloc[:, 3 : -1]
    targets.replace("M", 1, inplace = True) # malignant tumors have a value of 1
    targets.replace("B", 0, inplace = True) # benign tumors have a value of 0
    network_input = network_input.to_numpy()
    targets = targets.to_numpy()

    targets_percent = np.zeros((targets.shape[0], 2))
    for i, t in enumerate(targets):
        targets_percent[i, t] = 1

    # splitting and scaling: 
    train_in, test_in, train_o, test_o = train_test_split(network_input, targets_percent, test_size = 0.2)
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # testing out different scalers
    std_train_in = standard_scaler.fit_transform(train_in)
    std_test_in = standard_scaler.fit(test_in)
    mm_train_in = minmax_scaler.fit_transform(train_in)
    mm_test_in = minmax_scaler.fit_transform(test_in)

    train_in = std_train_in
    test_in = std_test_in

    # test how standarized data looks: 
    print(f"train in:\n{train_in[:10,:10]}\ntest in:\n{test_in[:10,:10]}")

    # not scaling targets as they are either 0 or 1 and should stay that way. 
    # checking for nan in matrix: 
    assert True not in np.isnan(network_input), "WARNING: NaN in network input matrix"
    print("no NaN's found in design matrix")

    # display dimensions
    print(f"INPUT MATRIX SHAPE: \n{network_input.shape}\nTARGET SHAPE: {targets.shape}")

    # making the neural network object from homemade NN class
    layer_output_sizes = [40, 2]
    activators = [sigmoid, softmax]
    print(f"\n\n ------------ Building neural network with {len(layer_output_sizes)} layers ------------ \n")
    NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, cost_function = binary_cross_entropy)
    print(f"Predicting test set with randomly initialized layers and weights:\n {NN.predict(test_in)[:10, :10]}")
    print("And train set:\n", NN.predict(train_in)[:10, :10])
    
    # training neural network
    epochs = 1000
    lrate = 0.01
    print(f"\n\n ------------ Training network with {epochs} epochs and {lrate} learning rate ------------\n")
    NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True)
    print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
    print("And some of train set:\n", NN.predict(train_in)[:10, :10])





main()

