
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ptwo.models import NeuralNetwork
from ptwo.costfuns import binary_cross_entropy
from ptwo.optimizers import ADAM
from ptwo.plot import set_plt_params
from ptwo.activators import relu6, sigmoid


set_plt_params()

epochs = 60
layer_output_sizes = [100,  2]
activators1 = [sigmoid, sigmoid]
activators2 = [relu6, sigmoid]
data_title = "ADAM"
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

# testing out different scalers
std_train_in = standard_scaler.fit_transform(train_in)
std_test_in = standard_scaler.transform(test_in)
train_in = std_train_in
test_in = std_test_in

print("\n----------------------------------------------------------------------------------------")
print("[                    Exploring NN with SGD and ADAM + autodiff                         ] ")
print("----------------------------------------------------------------------------------------\n")

fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
for activators in [activators1, activators2]:
    print(f"\n\n ------------ Training network with {epochs} epochs and {round(0.1, 4)} learning rate ------------\n")
    np.random.seed(42)
    NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, optimizer = ADAM(), cost_function = binary_cross_entropy, classification = True)
    NN.kfold_train(train_in, train_o, epochs = epochs, learning_rate = 0.1, verbose = True, batch_size = 150, k = 10)
    #print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
    #print("And some of train set:\n", NN.predict(train_in)[:10, :10])

    # plotting per learning rate
    ax1.plot(NN.cost_evolution, label = f"activation: {[f.__name__ for f in activators]}") #range(0, epochs, 100)
    ax2.plot(NN.accuracy_evolution, label = f"activation: {[f.__name__ for f in activators]}")

ax1.set_title("Loss")
ax1.set_ylabel("Binary cross entropy")
ax1.set_xlabel("Per 10 epochs")
ax1.legend(loc= "upper right")
plt.tight_layout()
plt.savefig("./latex/figures/adam_sgd_costLR.pdf", bbox_inches = "tight")
#plt.show()

ax2.set_title("Accuracy")
ax2.set_xlabel("Per 10 epochs")
ax2.set_ylabel("Accuracy")
ax2.axhline(1, ls = ":")
ax2.legend(loc= "upper right")
plt.tight_layout()
plt.savefig("./latex/figures/THIS_adam_sgd_LR01-sigmoid+sigmoid_relu6+sigmoid.pdf", bbox_inches = "tight")
plt.show()