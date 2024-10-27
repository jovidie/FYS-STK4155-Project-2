from src.ptwo.NeuralNetwork import NeuralNetwork

def test():

    np.random.seed(42)
    network_input_size = 4
    layer_output_sizes = [12, 10, 3]
    activation_funcs = [ReLU, ReLU, sigmoid]
    layers = create_layers_batch(network_input_size, layer_output_sizes)
    
    NN = NeuralNetwork(network_input, targets, layer_output_sizes, activation_funcs)
