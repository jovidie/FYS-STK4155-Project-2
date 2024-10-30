print("TEST RUNNING")

from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, ReLU
from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
# maco


def main():

    def cost_binary_cross_entropy(predict, target):
        """
        Loss function used in binary classification when target variable has two possible 
        outcomes: 1 or 0, 
        Args: 
        - predict is the prediction we have from input
        - target are the targets we know to match input
    """
        return np.mean((predict*np.log(target)) + ((1 - predict) * np.log(1 - target)))
    
    def mse(predict, target):
        return np.mean((predict - target) ** 2)


    # data prepping: 
    print("PREPPING DATA")
    terrain_full = imread('../data/SRTM_data_Norway_2.tif')
    terrain1 = terrain_full[1050:1250, 500:700]
    x_1d = np.arange(terrain1.shape[1])
    y_1d = np.arange(terrain1.shape[0])
    x_2d, y_2d = np.meshgrid(x_1d,y_1d)
    xy = np.column_stack((x_2d.flatten(), y_2d.flatten()))
    targets = terrain1.flatten()

    print("PREPPING NETWORK")
    # network prepping: 
    network_input = xy
    network_input_size = xy.shape[1]
    layer_output_sizes = [12, 10, 1]
    activation_funcs = [ReLU, ReLU, lambda x: x]
    NN = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs=activation_funcs, cost_function = mse)

    print("TESTING NETWORK")
    print(NN.predict(network_input).shape)
    plt.imshow(NN.predictions.reshape(200, 200), cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    print("TRAINING NETWORK")
    NN.train_network(network_input, targets)
    print(NN.predict(network_input).shape)
    plt.imshow(NN.predictions.reshape(200, 200), cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

main()
print("FINISHED TEST")
