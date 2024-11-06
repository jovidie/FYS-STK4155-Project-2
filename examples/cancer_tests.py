
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
from ptwo.optimizers import RMSProp, Momentum, AdaGrad, ADAM
from ptwo.plot import set_plt_params

set_plt_params()


def main():
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
    minmax_scaler = MinMaxScaler()

    # testing out different scalers
    std_train_in = standard_scaler.fit_transform(train_in)
    std_test_in = standard_scaler.transform(test_in)
    mm_train_in = minmax_scaler.fit_transform(train_in)
    mm_test_in = minmax_scaler.transform(test_in)

    train_in = std_train_in
    test_in = std_test_in

    # test how standarized data looks: 
    # print(f"train in:\n{train_in}\ntest in:\n{test_in}")

    # not scaling targets as they are either 0 or 1 and should stay that way. 
    # checking for nan in matrix: 
    assert True not in np.isnan(network_input), "WARNING: NaN in network input matrix"
    print("no NaN's found in design matrix")

    # display dimensions
    print(f"INPUT MATRIX SHAPE: \n{network_input.shape}\nTARGET SHAPE: {targets.shape}")

    # making the neural network object from homemade NN class
    layer_output_sizes = [50, 10, 2]
    activators = [sigmoid, sigmoid, softmax]
    print(f"\n\n ------------ Building neural network with {len(layer_output_sizes)} layers ------------ \n")
    NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, cost_function = binary_cross_entropy)
    print(f"Predicting test set with randomly initialized layers and weights:\n {NN.predict(test_in)[:10, :10]}")
    print("And train set:\n", NN.predict(train_in)[:10, :10])
    #fig, axs = plt.subplots(2, 1)
    

#-------------------------------------------------------------------------------------------------------------------------------
# GD, no optimizer
    epochs = 500
    lrates = np.linspace(0, 0.9, 10)

    def gd_test(fig, axs, epochs = 500, lrates = np.linspace(0, 0.9, 10), display_data = True):

        print("\n----------------------------------------------------------------------------------------")
        print("[  Exploring NN with GD and automatic differentiation across different learning rates   ] ")
        print("----------------------------------------------------------------------------------------\n")

        for lrate in lrates: 
            print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
            np.random.seed(42)
            NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, cost_function = binary_cross_entropy, classification = True)
            NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True)
            #print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
            #print("And some of train set:\n", NN.predict(train_in)[:10, :10])

            # plotting per learning rate
            axs[0].plot(NN.cost_evolution, label = f"GD; no optimizer, lr: {round(lrate, 4)}") #range(0, epochs, 100)
            axs[1].plot(NN.accuracy_evolution, label = f"GD; no optimmizer, lr: {round(lrate, 4)}")

        if display_data:
            axs[0].set_title("Loss")
            axs[0].set_ylabel("Binary cross entropy")
            axs[0].set_xlabel("Epochs")
            axs[1].set_title("Accuracy")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Accuracy")
            axs[1].axhline(1, ls = ":")
            axs[0].legend()
            axs[1].legend()
            fig.suptitle(f"Loss function and accuracy, GD w/{epochs} epochs - no gradient optimizer")
            plt.savefig("./latex/figures/gd_autodiff_w_learningrates.pdf", bbox_inches = "tight")
            plt.show()
    
    #gd_test(fig, axs)

#-------------------------------------------------------------------------------------------------------------------------------
# SGD, no optimizer
    epochs = 500
    lrates = np.linspace(0, 0.9, 10)

    def sgd_test(fig, axs, epochs = 500, lrates = np.linspace(0, 0.9, 10), display_data = True):
        # training neural network

        print("\n----------------------------------------------------------------------------------------")
        print("[  Exploring NN with SGD and automatic differentiation across different learning rates  ] ")
        print("----------------------------------------------------------------------------------------\n")

        for lrate in lrates: 
            print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
            np.random.seed(42)
            NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, cost_function = binary_cross_entropy, classification = True)
            NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True, batch_size = 150)
            #print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
            #print("And some of train set:\n", NN.predict(train_in)[:10, :10])

            # plotting per learning rate
            axs[0].plot(NN.cost_evolution, label = f"SGD; no optimizer, lr: {round(lrate,2)}") #range(0, epochs, 100)
            axs[1].plot(NN.accuracy_evolution, label = f"SGD; no optimizer, lr: {round(lrate, 4)}")

        if display_data:
            axs[0].set_title("Loss")
            axs[0].set_ylabel("Binary cross entropy")
            axs[0].set_xlabel("Epochs")
            axs[1].set_title("Accuracy")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Accuracy")
            axs[1].axhline(1, ls = ":")
            axs[0].legend()
            axs[1].legend()
            fig.suptitle(f"Loss function and accuracy, SGD w/{epochs} epochs - no gradient optimizer")
            plt.savefig("./latex/figures/sgd_autodiff_w_learningrates.pdf", bbox_inches = "tight")
            plt.show()

    #sgd_test(fig, axs)

#-------------------------------------------------------------------------------------------------------------------------------
# SGD with RMSprop

    def rmsprop_sgd_test(fig, axs, epochs = 500, lrates = [0.1], display_data = True):
        # training neural network
        rho = 0.99

        print("\n----------------------------------------------------------------------------------------")
        print("[                   Exploring NN with SGD and RMSProp + autodiff                     ] ")
        print("----------------------------------------------------------------------------------------\n")

        for lrate in lrates: 
            print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
            np.random.seed(42)
            NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, optimizer = RMSProp(rho), cost_function = binary_cross_entropy, classification = True)
            NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True, batch_size = 150)
            #print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
            #print("And some of train set:\n", NN.predict(train_in)[:10, :10])

            # plotting per learning rate
            axs[0].plot(NN.cost_evolution, label = f"SGD; RMSProp, lr: {round(lrate,2)}") #range(0, epochs, 100)
            axs[1].plot(NN.accuracy_evolution, label = f"SGD; RMSprop, lr: {round(lrate, 4)}")

        if display_data:
            axs[0].set_title("Loss")
            axs[0].set_ylabel("Binary cross entropy")
            axs[0].set_xlabel("Epochs")
            axs[1].set_title("Accuracy")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Accuracy")
            axs[1].axhline(1, ls = ":")
            axs[0].legend()
            axs[1].legend()
            fig.suptitle(f"Loss function and accuracy, RMSProp SGD w/{epochs} epochs - no gradient optimizer")
            plt.savefig("./latex/figures/rmsprop_sgd_autodiff_w_learningrates.pdf", bbox_inches = "tight")
            plt.show()

    #rmsprop_sgd_test(fig, axs)

#-------------------------------------------------------------------------------------------------------------------------------
# SGD with Momentum

    def momentum_sgd_test(fig, axs, epochs = 500, lrates = [0.1], display_data = True):
        # training neural network

        print("\n----------------------------------------------------------------------------------------")
        print("[                  Exploring NN with SGD and Momentum + autodiff                        ] ")
        print("----------------------------------------------------------------------------------------\n")

        for lrate in lrates: 
            print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
            np.random.seed(42)
            NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, optimizer = Momentum(), cost_function = binary_cross_entropy, classification = True)
            NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True, batch_size = 150)
            #print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
            #print("And some of train set:\n", NN.predict(train_in)[:10, :10])

            # plotting per learning rate
            axs[0].plot(NN.cost_evolution, label = f"SGD; Momentum, lr: {round(lrate,2)}") #range(0, epochs, 100)
            axs[1].plot(NN.accuracy_evolution, label = f"SGD; Momentum, lr: {round(lrate, 4)}")

        if display_data:
            axs[0].set_title("Loss")
            axs[0].set_ylabel("Binary cross entropy")
            axs[0].set_xlabel("Epochs")
            axs[1].set_title("Accuracy")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Accuracy")
            axs[1].axhline(1, ls = ":")
            axs[0].legend()
            axs[1].legend()
            fig.suptitle(f"Loss function and accuracy, Momentum SGD w/{epochs} epochs - no gradient optimizer")
            plt.savefig("./latex/figures/momentum_sgd_autodiff_w_learningrates.pdf", bbox_inches = "tight")
            plt.show()

    #momentum_sgd_test(fig, axs)

#-------------------------------------------------------------------------------------------------------------------------------
# SGD with AdaGrad

    def adagrad_sgd_test(fig, axs, epochs = 500, lrates = [0.1], display_data = True):
        # training neural network

        print("\n----------------------------------------------------------------------------------------")
        print("[                 Exploring NN with SGD and AdaGrad + autodiff                         ] ")
        print("----------------------------------------------------------------------------------------\n")

        for lrate in lrates: 
            print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
            np.random.seed(42)
            NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, optimizer = AdaGrad(), cost_function = binary_cross_entropy, classification = True)
            NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True, batch_size = 150)
            #print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
            #print("And some of train set:\n", NN.predict(train_in)[:10, :10])

            # plotting per learning rate
            axs[0].plot(NN.cost_evolution, label = f"SGD; AdaGrad, lr: {round(lrate,2)}") #range(0, epochs, 100)
            axs[1].plot(NN.accuracy_evolution, label = f"SGD; AdaGrad, lr: {round(lrate, 4)}")

        if display_data:
            axs[0].set_title("Loss")
            axs[0].set_ylabel("Binary cross entropy")
            axs[0].set_xlabel("Epochs")
            axs[1].set_title("Accuracy")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Accuracy")
            axs[1].axhline(1, ls = ":")
            axs[0].legend()
            axs[1].legend()
            fig.suptitle(f"Loss function and accuracy, AdaGrad SGD w/{epochs} epochs - no gradient optimizer")
            plt.savefig("./latex/figures/adagrad_sgd_autodiff_w_learningrates.pdf", bbox_inches = "tight")
            plt.show()

    #adagrad_sgd_test(fig, axs)

#-------------------------------------------------------------------------------------------------------------------------------
# SGD with ADAM

    def adam_sgd_test(fig, axs, epochs = 500, lrates = [0.1], display_data = True):
        # training neural network

        print("\n----------------------------------------------------------------------------------------")
        print("[                    Exploring NN with SGD and ADAM + autodiff                         ] ")
        print("----------------------------------------------------------------------------------------\n")

        for lrate in lrates: 
            print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
            np.random.seed(42)
            NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, optimizer = ADAM(), cost_function = binary_cross_entropy, classification = True)
            NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True, batch_size = 150)
            #print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
            #print("And some of train set:\n", NN.predict(train_in)[:10, :10])

            # plotting per learning rate
            axs[0].plot(NN.cost_evolution, label = f"SGD; ADAM, lr: {round(lrate,2)}") #range(0, epochs, 100)
            axs[1].plot(NN.accuracy_evolution, label = f"SGD; ADAM, lr: {round(lrate, 4)}")

        if display_data:
            axs[0].set_title("Loss")
            axs[0].set_ylabel("Binary cross entropy")
            axs[0].set_xlabel("Epochs")
            axs[1].set_title("Accuracy")
            axs[1].set_xlabel("Per 100 epochs")
            axs[1].set_ylabel("Accuracy")
            axs[1].axhline(1, ls = ":")
            axs[0].legend()
            axs[1].legend()
            fig.suptitle(f"Loss function and accuracy, ADAM SGD w/{epochs} epochs")
            fig.tight_layout()
            plt.savefig("./latex/figures/adam_sgd_autodiff_w_learningrates.pdf", bbox_inches = "tight")
            plt.show()

    #adam_sgd_test(fig, axs)

    all_tests = [gd_test, sgd_test, momentum_sgd_test, rmsprop_sgd_test, adagrad_sgd_test, adam_sgd_test]
    l_rate = [[x] for x in [0.9, 0.9, 0.1, 0.1, 0.1, 0.1]]
    """
    for ind, test in enumerate(all_tests):
        test(fig, axs, epochs = 400, lrates = l_rate[ind], display_data = False)
    #    if ind == len(all_tests)-1:
    #        test(fig, axs, epochs = 500, lrates = [0.9], display_data = True
    axs[0].set_title("Loss")
    axs[0].set_ylabel("Binary cross entropy")
    axs[0].set_xlabel("Epochs")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Per 100 Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].axhline(1, ls = ":")
    axs[0].legend(loc = "upper right")
    axs[1].legend(loc = "upper right")
    fig.suptitle(f"Loss function and accuracy wih different optimizsers")
    fig.tight_layout()
    """
    #plt.savefig("./latex/figures/all_methods_cancer_NN.pdf", bbox_inches = "tight")
    #plt.show()

    #fig, axis = plt.subplots(2,1)
    #fig.suptitle(f"Loss function and accuracy for different ADAM learning rates")
    #adam_sgd_test(fig, axis, lrates = 10.**np.arange(-5,0))

    print("\n----------------------------------------------------------------------------------------")
    print("[                    Exploring NN with SGD and ADAM + autodiff                         ] ")
    print("----------------------------------------------------------------------------------------\n")

    fig1, ax1 = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    for lrate in 10.**np.arange(-5,1):
        print(f"\n\n ------------ Training network with {epochs} epochs and {round(lrate, 4)} learning rate ------------\n")
        np.random.seed(42)
        NN = NeuralNetwork(network_input_size = network_input.shape[1], layer_output_sizes = layer_output_sizes, activation_funcs = activators, optimizer = ADAM(), cost_function = binary_cross_entropy, classification = True)
        NN.train_network(train_in, train_o, epochs = epochs, learning_rate = lrate, verbose = True, batch_size = 150)
        #print(f"Predicting on some of test set after traning:\n {NN.predict(test_in)[:10, :10]}")
        #print("And some of train set:\n", NN.predict(train_in)[:10, :10])

        # plotting per learning rate
        ax1.plot(NN.cost_evolution, label = f"lr: {round(lrate,4)}") #range(0, epochs, 100)
        ax2.plot(NN.accuracy_evolution, label = f"lr: {round(lrate, 4)}")
    
    ax1.set_title("Loss")
    ax1.set_ylabel("Binary cross entropy")
    ax1.set_xlabel("Epochs")
    ax1.legend(loc= "upper right")
    plt.tight_layout()
    plt.savefig("./latex/figures/adam_sgd_costLR.pdf", bbox_inches = "tight")
    #plt.show()

    ax2.set_title("Accuracy")
    ax2.set_xlabel("Per 100 epochs")
    ax2.set_ylabel("Accuracy")
    ax2.axhline(1, ls = ":")
    ax2.legend(loc= "upper right")
    plt.tight_layout()
    plt.savefig("./latex/figures/adam_sgd_costLR.pdf", bbox_inches = "tight")
    plt.show()

main()


