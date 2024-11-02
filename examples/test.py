print("TEST RUNNING")

from ptwo.models import NeuralNetwork
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ptwo.activators import sigmoid, ReLU
from ptwo.optimizers import ADAM
from imageio import imread
import autograd.numpy as np
import matplotlib.pyplot as plt
from ptwo.costfuns import mse, mse_der

def main():
    fig, axs = plt.subplots(1, 3)
    # data prepping: 
    print("PREPPING DATA")
    terrain_full = imread('./data/SRTM_data_Norway_2.tif')
    terrain1 = terrain_full[1050:1100, 500:550]
    axs[0].imshow(terrain1, cmap='gray')
    axs[0].set_title("Targets")

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
    layer_output_sizes = [int(targets.shape[0]/0.8), 10,  1]
    activation_funcs = [sigmoid, sigmoid, lambda x: x]
    NN = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, mse, target_means = np.mean(targets))

    print("Checking if cost-function is allright")
    print("MSE before training:", NN.get_cost(network_input, targets))
    print("Allright!")
    NN.predict(network_input)

    print("TESTING NETWORK")
    axs[1].imshow(NN.predictions.reshape(terrain1.shape), cmap='gray')
    axs[1].set_title("FFNN before training")

    print("TRAINING NETWORK")
    NN.train_network(network_input, targets, learning_rate=0.01, epochs=1000, verbose = True)
    NN.predict(network_input)
    axs[2].imshow(NN.predictions.reshape(terrain1.shape), cmap='gray')
    axs[2].set_title("FFNN after training")
    fig.supxlabel('X')
    fig.supylabel('Y')
    fig.suptitle('Performance on FFNN, MSE cost function')
    plt.show()

main()
print("FINISHED TEST")
