from utilities import load_data, normalize, one_hot_encode
from network import NeuralNetwork

import numpy as np

# load data
x_train, y_train, x_test, y_test = load_data()

# preprocesses data
x_train = normalize(x_train)
x_test = normalize(x_test)

y_train_onehot = one_hot_encode(y_train)
y_test_onehot = one_hot_encode(y_test)

# Set network size
input_nodes = 784
hidden_nodes = 300
output_nodes = 10
learning_rate = 0.1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Train the network
epochs = 10
for e in range(epochs):
    # Shuffle training data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    for i in indices:
        inputs = x_train[i]
        targets = y_train_onehot[i]
        nn.train(inputs, targets)
    print(f"Epoch {e+1}/{epochs} complete.")

# Test the network
scorecard = []
for i in range(len(x_test)):
    correct_label = y_test[i]
    outputs = nn.query(x_test[i])  # outputs: (10,1)
    # the index of the highest value corresponds to the predicted label
    label = np.argmax(outputs)
    # Check if correct
    scorecard.append(1 if (label == correct_label) else 0)

accuracy = np.mean(scorecard)
print("Test Accuracy:", accuracy)
