from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ptwo.models import NeuralNetwork
from ptwo.activators import sigmoid, ReLU, softmax
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

clf = MLPClassifier(random_state=1, max_iter=500).fit(train_in, train_o)

clf.predict_proba(test_in[:1])
clf.predict(test_in[:5, :])
clf.score(test_in, test_o)