from sklearn.neural_network import MLPClassifier
import pandas as pd
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, ReLU, softmax, leaky_ReLU, relu
from ptwo.costfuns import binary_cross_entropy
from ptwo.optimizers import RMSProp, Momentum, AdaGrad, ADAM

#importing data:
data = pd.read_csv("./data/wisconsin_breast_cancer_data.csv")
print(f"DATA\n{data}")

# preprocessing data: 
targets = data["diagnosis"]
network_input = data.iloc[:, 3 : -1]
targets.replace("M", 1, inplace = True) # malignant tumors have a value of 1
targets.replace("B", 0, inplace = True) # benign tumors have a value of 0
total_Ms = np.sum(targets)
network_input = network_input.to_numpy()
targets = targets.to_numpy()

print(f"\n ---- {total_Ms} malignant tumors \n ---- {int(len(targets) - total_Ms)} benign tumors\n")

targets_percent = np.zeros((targets.shape[0], 2))
for i, t in enumerate(targets):
    targets_percent[i, t] = 1

# splitting and scaling: 
train_in, test_in, train_o, test_o = train_test_split(network_input, targets_percent, test_size = 0.2)
standard_scaler = StandardScaler()
#minmax_scaler = MinMaxScaler()
train_in = standard_scaler.fit_transform(train_in)
test_in = standard_scaler.transform(test_in)
#train_in = minmax_scaler.fit_transform(train_in)
#test_in = minmax_scaler.transform(test_in)

#joint parameters
epochs = 500
lrate = 0.1
layer_output_sizes = [100, 50,  2]
activators = [sigmoid, sigmoid, sigmoid]

print(f"\n\n ------------ Building neural networks with {len(layer_output_sizes)} layer for Sklearn and own FFNN ------------ \n")
#sklearn classifier: 
clf = MLPClassifier(random_state = 1, max_iter = epochs, learning_rate_init = lrate)
clf.fit(train_in, train_o)
clf.predict_proba(test_in[:1])
clf.predict(test_in[:5, :])
clf.score(test_in, test_o)

#our classifier
print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
np.random.seed(42)
NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, optimizer = ADAM(), cost_function = binary_cross_entropy, classification = True)
NN.predict(train_in)
#print(NN.predictions)
NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True, batch_size = 150)