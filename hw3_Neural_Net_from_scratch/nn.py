import numpy as np
import time, sys

NODES_PER_LAYER = [5,9]
ALPHA = 0.05
MINI_BATCH_SIZE = 10

def linebreak(label="*"):
    print("*" * 12, label, "*" * 12)

def load_training_data(filename):
    return np.loadtxt(filename, delimiter=",", dtype=float)

def load_training_labels(filename):
    return np.loadtxt(filename, dtype=int)

def load_test_data(filename):
    return np.loadtxt(filename, delimiter="," , dtype=float)

def load_test_labels(filename):
    return np.loadtxt(filename, dtype=int)

def write_predictions(predictions):
    np.savetxt('test_predictions.csv', predictions, fmt="%d", delimiter='\n')

def sigmoid(x):
    x = np.clip(x, -900, 900)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1-s)

def loss(output, target):
    return - (target * np.log(output) + (1 - target) * np.log(1 - output))

def loss_prime(output, target):
    target += 0.00000001
    output += 0.00000001
    return - target/output + (1-target)/(1-output)

def relu(num):
    return np.where(num >= 0, num, num * 0.1)
    
def relu_prime(num):
    return np.where(num < 0, 0.1, 1.0)

def get_predictions(network, test_data):
    predictions = []
    for i in range(len(test_data)):
        x1, x2 = test_data[i]
        zs, activations = network.feed_forward([x1,x2,x1**2, x2**2, x1*x2])
        output = activations[-1]
        if output >= 0.5:
            output = 1
        else:
            output = 0        
        predictions.append(output)
    return predictions

def test_model(network, test_data, test_labels, verbose=False):
    correct = 0
    for i in range(len(test_data)):
        x1, x2 = test_data[i]
        zs, activations = network.feed_forward([x1,x2,x1**2, x2**2, x1*x2])
        output = activations[-1]
        if output >= 0.5:
            output = 1
        else:
            output = 0
        if output == test_labels[i]:
            correct += 1
        if verbose:
            print(f"{zs[-1]} | {activations[-1]} \t({output} = {test_labels[i]})")

    print(f"Accuracy: {correct} of {len(test_data)} :: {correct/len(test_data)}")

class NeuralNetwork():
    def __init__(self, shape) -> None:
        self.weights = self.randomize_basic(shape)
        self.biases = self.init_biases(shape)
        self.depth = len(self.weights)

    def print_weights(self):
        for w in self.weights:
            linebreak()
            print(w)
    
    def print_biases(self):
        for b in self.biases:
            linebreak()
            print(b)

    def randomize_basic(self, shape):
        layers = []
        for i in range(len(shape)-1):
            layers.append(2*np.random.rand(shape[i+1], shape[i])-1)
        layers.append(2*np.random.rand(1,shape[-1])-1)
        return layers

    def init_biases(self, shape):
        layers = []
        for i in range(1, len(shape)):
            layers.append(np.zeros((1, shape[i])))
        layers.append(np.zeros((1,1)))
        return layers

    def feed_forward_preserved(self, x):
        zs = []
        activations = []
        a = np.array([x]).T
        activations.append(a)
        for i in range(self.depth-1):
            z = np.dot(self.weights[i], a)
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        z = np.dot(self.weights[self.depth-1], a)
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)
        print("zs")
        print(zs)

        return zs, activations

    def feed_forward(self, x):
        zs = []
        activations = []
        a = np.array([x]).T
        activations.append(a)
        z = np.dot(self.weights[0], a)
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)
        for i in range(1, self.depth):
            z = np.dot(self.weights[i], a) + self.biases[i].T
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        return zs, activations

    def back_propogation(self, x, target):
        adjustment = [0 for i in range(self.depth)]
        adj_biases = [0 for i in range(self.depth)]

        z, a = self.feed_forward(x)

        error = loss_prime(a[-1], target)

        delta = np.multiply(error, sigmoid_prime(z[-1]))

        adjustment[-1] = np.multiply(a[-2],delta)
        adj_biases[-1] = delta

        for i in range(2,self.depth):
            sig = sigmoid_prime(z[-i])
            delta = np.multiply(np.dot(self.weights[-i+1].T, delta), sig)
            adjustment[-i] = np.dot(a[-i-1], delta.T)
            adj_biases[-i] = delta

        sig = sigmoid_prime(z[-self.depth])
        delta = np.multiply(np.dot(self.weights[-self.depth+1].T, delta), sig)
        adjustment[-self.depth] = np.dot(a[-self.depth-1], delta.T)
        adj_biases[-self.depth] = delta

        return adjustment, adj_biases

    def learn(self, adjustments, adj_bias, n):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - np.multiply(ALPHA, adjustments[i].T) / n
            self.weights[i] = np.clip(self.weights[i], -10.0, 10.0)
            self.biases[i] = self.biases[i] - np.multiply(ALPHA, adj_bias[i].T) / n
            self.biases[i] = np.clip(self.biases[i], -10.0, 10.0)


    def get_output(self, x):
        z, a = self.feed_forward(x)
        if a[-1] >= 0.5:
            return 1
        else:
            return 0

def main():
    begin1 = time.time()
    # Get Data and path
    # data_set = "spiral"
    """ For testing """
    data_set = sys.argv[1]
    training_data = load_training_data(f"{data_set}_train_data.csv")
    training_labels = load_training_labels(f"{data_set}_train_label.csv")
    test_data = load_test_data(f"{data_set}_test_data.csv")
    test_labels = load_test_labels(f"{data_set}_test_label.csv")
    print("-" * 20, data_set, NODES_PER_LAYER, "-" * 20)
    print("-" * 20, f"Alpha: {ALPHA}", "-" * 20)

    """ For submission """
    # training_data = load_training_data(sys.argv[1])
    # training_labels = load_training_labels(sys.argv[2])
    # test_data = load_test_data(sys.argv[3])

    network = NeuralNetwork(NODES_PER_LAYER)

    # network.print_weights()
    # network.print_biases()

    n = len(training_data)
    epochs = 1000
    total_iterations = n * epochs
    e=0
    avg_adj = [np.array([]) for i in range(network.depth)]
    avg_bias = [np.array([]) for i in range(network.depth)]

    # x1, x2 = training_data[0]
    # z, a = network.feed_forward([x1,x2,x1**2,x2**2,x1*x2])
    # print(a)

    for j in range(epochs):
        for i in range(n):
            # print(i, i % MINI_BATCH_SIZE)
            x1, x2 = training_data[i]
            adj, adj_biases = network.back_propogation([x1,x2,x1**2, x2**2, x1*x2], training_labels[i])
            # print(adj)
            # network.learn(adj, 1.0)
            for k, a in enumerate(adj):
                if avg_adj[k].shape == a.shape:
                    # print(avg_adj[i].shape, a.shape)
                    avg_adj[k] += a
                    avg_bias[k] += adj_biases[k]
                else:
                    avg_adj[k] = a
                    avg_bias[k] = adj_biases[k]
                    
            # print (i % MINI_BATCH_SIZE)
            if (i+1) % MINI_BATCH_SIZE == 0:
                # print("learning...")
                network.learn(avg_adj, avg_bias, MINI_BATCH_SIZE)
                avg_adj = [np.array([]) for p in range(network.depth)]
        e += 1
        if time.time() - begin1 > 110.0:
            break

    # network.print_weights()
    # network.print_biases()

    """ For Testing """
    test_model(network, test_data, test_labels, verbose=False)
    
    """ For submission """
    # predictions = get_predictions(network, test_data)
    # write_predictions(predictions)
    
    print(f"Execution time: {time.time() - begin1}")
    print(f"Epochs: {e}")

    # print("\n\n")
    # network.print_weights()      


if __name__ == "__main__":
    
    main()