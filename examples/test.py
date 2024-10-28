from src.ptwo.models import NeuralNetwork
from src.ptwo.activators import sigmoid, ReLU
from imageio import imread
import numpy as np

def test():
    # data prepping: 
    terrain_full = imread('../data/SRTM_data_Norway_2.tif')
    terrain1 = terrain_full[1050:1250, 500:700]
    x_1d = np.arange(terrain1.shape[1])
    y_1d = np.arange(terrain1.shape[0])
    x_2d, y_2d = np.meshgrid(x_1d,y_1d)
    xy = np.column_stack((x_2d.flatten(), y_2d.flatten()))
    targets = terrain1.flatten()

    # network prepping: 
    network_input = xy
    layer_output_sizes = [12, 10, 1]
    activation_funcs = [ReLU, ReLU, sigmoid]

    NN = NeuralNetwork(network_input, targets, layer_output_sizes, activation_funcs)
    
