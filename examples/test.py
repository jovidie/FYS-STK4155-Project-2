print("TEST RUNNING")

from ptwo.models import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from ptwo.activators import sigmoid, ReLU
from ptwo.optimizers import ADAM
from imageio import imread
import autograd.numpy as np
import matplotlib.pyplot as plt
from ptwo.costfuns import mse



def main():

    def binary_cross_entropy(predict, target):
        """
        Loss function used in binary classification when target variable has two possible 
        outcomes: 1 or 0, 
        Args: 
        - predict is the prediction we have from input
        - target are the targets we know to match input
    """
        return - np.mean(target * np.log(predict) + (1 - target) * np.log(1 - predict))

    # data prepping: 
    print("PREPPING DATA")
    terrain_full = imread('./data/SRTM_data_Norway_2.tif')
    terrain1 = terrain_full[1050:1250, 500:700]
    plt.imshow(terrain1, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # this is maybe not neccessary 
    x_1d = np.arange(terrain1.shape[1])
    y_1d = np.arange(terrain1.shape[0])
    x_2d, y_2d = np.meshgrid(x_1d,y_1d)
    xy = np.column_stack((x_2d.flatten(), y_2d.flatten()))
    targets = terrain1.flatten()

    scaler = StandardScaler()
    xy = scaler.fit_transform(xy)
    targets = scaler.fit_transform(targets.reshape(-1,1))

    print("PREPPING NETWORK")
    # network prepping: 
    network_input = xy
    network_input_size = xy.shape[1]
    layer_output_sizes = [5, 10, 15, 1]
    activation_funcs = [ReLU, ReLU, sigmoid,lambda x: x]
    NN = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, mse, optimizer=ADAM())

    print("Checking if cost-function is allright")
    print("MSE before training:", NN.get_cost(network_input, targets))
    print("Allright!")
    NN.predict(network_input)

    print("TESTING NETWORK")
    plt.imshow(NN.predictions.reshape(200, 200), cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    print("TRAINING NETWORK")
    NN.train_network(network_input, targets, learning_rate=0.01, epochs=10000)
    plt.imshow(NN.predictions.reshape(200, 200), cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

main()
print("FINISHED TEST")
