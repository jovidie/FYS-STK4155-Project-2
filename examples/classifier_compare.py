from sklearn.neural_network import MLPClassifier
import pandas as pd
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, relu6
from ptwo.costfuns import binary_cross_entropy
from ptwo.optimizers import ADAM
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from ptwo.plot import set_plt_params

# standard settings for plots
set_plt_params()

# importing data:
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

# checking if our data set is balanced
print(f"\n ---- {total_Ms} malignant tumors \n ---- {int(len(targets) - total_Ms)} benign tumors\n")

# making targets into two columns for each option (binary) "B" and "M"
targets_percent = np.zeros((targets.shape[0], 2))
for i, t in enumerate(targets):
    targets_percent[i, t] = 1

# setting seed for reproducibility
np.random.seed(42)

# splitting and scaling: 
train_in, test_in, train_o, test_o = train_test_split(network_input, targets_percent, test_size = 0.2)
standard_scaler = StandardScaler()
train_in = standard_scaler.fit_transform(train_in)
test_in = standard_scaler.transform(test_in)

#joint parameters
epochs = 200
lrate = 0.1
layer_output_sizes = [100,  2]
activators = [relu6, sigmoid]
data_title = "ADAM"
metadata_f = [f.__name__ for f in activators]
metadata_l = [str(l) for l in layer_output_sizes]
combined_metadata = [f + "-" + l for f, l in zip(metadata_f, metadata_l)]
for combmet in combined_metadata:
    data_title += "_" + combmet

# set this to true to view and save confusion matrix after training and prediction
confusion = False

# set this to true to view aroc: 
roc = False



print(f"\n\n ------------ Building neural networks with {len(layer_output_sizes)} layer for Sklearn and own FFNN ------------ \n")
#sklearn classifier: 
clf = MLPClassifier(hidden_layer_sizes = layer_output_sizes, max_iter = epochs, learning_rate_init = lrate)
clf.fit(train_in, train_o)
clf_test_pred = clf.predict(test_in)
clf.score(test_in, test_o)

#our classifier
print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, optimizer = ADAM(), cost_function = binary_cross_entropy, classification = True)
NN.predict(train_in)
#print(NN.predictions)
NN.kfold_train(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True, batch_size = 150, k = 10)
test_pred = NN.predict(test_in)

# making predictions back into vector of 0 and 1's 
predi = np.argmax(test_pred, axis=1)
clf_predictions = np.argmax(clf_test_pred, axis=1)
golden = np.argmax(test_o, axis=1)


if confusion: 

    #confusion matrices
    conf1 = confusion_matrix(golden, clf_predictions, labels = [0, 1])
    ConfusionMatrixDisplay(conf1).plot()
    plt.title(f"Confusion matrix, Sklearn's NN, acc: {round(accuracy_score(clf_predictions, golden), 4)}")
    plt.savefig("./latex/figures/sklearnWBC_final_" + data_title + ".pdf", bbox_inches = "tight")
    plt.show()

    conf2 = confusion_matrix(golden, predi, labels = [0, 1])
    ConfusionMatrixDisplay(conf2).plot()
    plt.title(f"Confusion matrix, our NN, acc: {round(accuracy_score(predi, golden), 4)}")
    plt.savefig("./latex/figures/ourWBC_final_"+ data_title + ".pdf", bbox_inches = "tight")
    plt.show()

if roc: 

