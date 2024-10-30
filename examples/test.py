print("TEST RUNNING")

from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, ReLU
from imageio import imread
import numpy as np
import matplotlib.pyplot as plt

def main():
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
    layer_output_sizes = [12, 10, 1]
    activation_funcs = [ReLU, ReLU, sigmoid]
    NN = NeuralNetwork(network_input, layer_output_sizes, activation_funcs)

    print("TESTING NETWORK")
    print(NN.predict(network_input).shape)
    plt.imshow(NN.predictions.reshape(200, 200), cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    NN.train_network(network_input, targets)
    print(NN.predict(network_input).shape)
    plt.imshow(NN.predictions.reshape(200, 200), cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

main()
print("FINISHED TEST")
"""
if __name__ == '__test_':
    main()


"""

