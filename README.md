# Introduction

This project demonstrates a simple neural network implemented from scratch in Python to classify handwritten digits from the MNIST dataset. The network uses the standard logistic sigmoid activation function and trains using backpropagation. 


## Project Structure

The project is organized into three main Python files, plus a README/documentation file:

```
project/
│
├─ README.md          # Documentation and instructions
├─ network.py         # Defines the NeuralNetwork class and its functionality
├─ mnist_utils.py     # Utility functions for loading, preprocessing, and encoding MNIST data
└─ train.py           # The main script to train and test the neural network

```



## Overview of Files

### 1. network.py
   Contains the `NeuralNetwork` class:
   - **Initialization:** Sets the network architecture (input, hidden, output layers), learning rate, and random weight initialization.
   - **Activation Function (Sigmoid):** Uses $σ(x) = \frac{1}{(1+e^{-x})}$ which outputs a value in (0,1).
   - **Forward Pass:** Computes hidden and final outputs.
   - **Backpropagation:** Computes gradients and updates weights.
   - **Query Method:** Feeds forward inputs through the network to produce predictions.

### 2. utilities.py
#### 1. Load MNIST:
   Uses `from mnist_utils import load_data` to load data from `data` file:
   - `x_train, y_train`: Training images and labels
   - `x_test, y_test`: Test images and labels

#### 2. Normalize:
   - Scale pixel values from `[0,255]` to `[0.01,1.0] `for stable training under sigmoid activation.

#### 3. One-Hot Encoding
   - Converts digit labels (0-9) to vectors like `[0.01, ..., 0.99, ...]` where the target class is near 1 and others near 0.
   - Ex: convert `3` into `[0.01, 0.01, 0.01, 0.99, 0.01, ...]`. 

### 3. train.py
   The main script that:
#### Loads and preprocesses the MNIST dataset.
```python
# load data
x_train, y_train, x_test, y_test = load_data()

# preprocesses data
x_train = normalize(x_train)
x_test = normalize(x_test)

y_train_onehot = one_hot_encode(y_train)
y_test_onehot = one_hot_encode(y_test)

```

#### Initializes the neural network.
```python
# Set network size
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
```

#### Trains the network for a specified number of epochs.
- Shuffle the training data for each epoch.
- For each training sample, call: `nn.train(inputs, targets)`
- This performs one forward pass and one backward pass, updating the weights.
- Choose how many times the entire training set is passed through the network.
```python
# Train the network
epochs = 1
for e in range(epochs):
    # Shuffle training data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    for i in indices:
        inputs = x_train[i]
        targets = y_train_onehot[i]
        nn.train(inputs, targets)
    print(f"Epoch {e+1}/{epochs} complete.")
```

#### Evaluates the trained model on the test set and prints the accuracy.
```python
# Test the network
scorecard = []
for i in range(len(x_test)):
    correct_label = y_test[i]
    outputs = nn.query(x_test[i])  # outputs: (10,1)
    # the index of the highest value corresponds to the predicted label
    label = np.argmax(outputs)
    # Check if correct
    scorecard.append(1 if (label == correct_label) else 0)
```


---

# Neural Network Architecture

## Layers
- **Input Layer:** MNIST images are 28x28 pixels, flattened into a 784-element input vector.
- **Hidden Layer:** A single hidden layer with a tunable number of neurons (e.g., 100).
- **Output Layer:** 10 output neurons, one per digit class (0 through 9).

## Weights
- `w_input_hidden`: Connects the input to the hidden layer (shape: `(hidden_nodes, input_nodes)`).
- `w_hidden_output`: Connects the hidden layer to the output layer (shape: `(output_nodes, hidden_nodes)`).

Weights are initialized randomly with a distribution scaled by the inverse square root of the layer size to promote stable training.

## Activation Function (Sigmoid)

We use the standard logistic sigmoid function:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Properties
- Outputs range between (0,1).
- Smooth and differentiable, which makes it suitable for gradient-based training.
  
### Derivative

$$\sigma'(x) = \sigma(x) \times (1 - \sigma(x))$$


This derivative is efficiently computed once `σ(x)` is known.
This derivative is computed in `activation_derivative()` within the `NeuralNetwork` class.

# Forward Pass
The forward pass computes the network's outputs given an input vector. It consists of:

## 1. For an input vector:
### 1. Input → Hidden:
   
   $$hidden\_inputs = w\_input\_hidden \cdot inputs$$
   $$hidden\_outputs = \sigma(hidden\_inputs)$$


### 2. Hidden → Output:

   $$final\_inputs = w\_hidden\_output \cdot hidden\_outputs$$

   $$final\_outputs = \sigma(final\_inputs)$$

The `final_outputs` represent the network's predictions (confidence for each digit).

# Backpropagation (Training Step)

## 1. Output Error
   
   $$output\_errors = targets - final\_outputs$$


## 2. Output Layer Gradient:

   $$output\_grad = output\_errors \times \sigma'(final\_inputs)$$
   

## 3. Backpropagate to Hidden Layer:
   
   $$hidden\_errors = w\_hidden\_output^T \cdot output\_grad$$

## 4. Hidden Layer Gradient:
   
$$   hidden\_grad = hidden\_errors \times \sigma'(hidden\_inputs)$$


## 5. Update Weights:
### For `w_hidden_output`:
 
$$ w\_hidden\_output \leftarrow w\_hidden\_output + \eta \cdot (output\_grad \cdot hidden\_outputs^T)$$

   
### For `w_input_hidden`:
 $$w\_input\_hidden \leftarrow w\_input\_hidden + \eta \cdot (hidden\_grad \cdot inputs^T)$$


- Here, $\eta$ is the learning rate, controlling how big the weight updates are.





