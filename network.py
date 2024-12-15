import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        # Initialize weights with small random values
        # shape: (hidden_nodes, input_nodes)
        self.w_input_hidden = np.random.normal(
            0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes)
        )
        # shape: (output_nodes, hidden_nodes)
        self.w_hidden_output = np.random.normal(
            0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes)
        )

        # Given custom activation function:
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        # Compute the activation once, then use it to find the derivative
        a = self.activation_function(x)
        return a * (1 - a)

    def forward_pass(self, inputs):
        # inputs: (input_nodes, 1)
        hidden_inputs = np.dot(self.w_input_hidden, inputs)  # (hidden_nodes, 1)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.w_hidden_output, hidden_outputs)  # (output_nodes, 1)
        final_outputs = self.activation_function(final_inputs)

        return hidden_inputs, hidden_outputs, final_inputs, final_outputs

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T  # (input_nodes, 1)
        targets = np.array(targets_list, ndmin=2).T  # (output_nodes, 1)

        # Forward pass
        hidden_inputs, hidden_outputs, final_inputs, final_outputs = self.forward_pass(
            inputs
        )

        # Compute output error
        output_errors = targets - final_outputs  # (output_nodes, 1)

        # Output layer gradient
        output_grad = output_errors * self.activation_derivative(
            final_inputs
        )  # (output_nodes, 1)

        # Hidden layer errors
        hidden_errors = np.dot(self.w_hidden_output.T, output_grad)  # (hidden_nodes,1)

        # Hidden layer gradient
        hidden_grad = hidden_errors * self.activation_derivative(
            hidden_inputs
        )  # (hidden_nodes,1)

        # Update the weights
        self.w_hidden_output += self.lr * np.dot(
            output_grad, hidden_outputs.T
        )  # (output_nodes, hidden_nodes)
        self.w_input_hidden += self.lr * np.dot(
            hidden_grad, inputs.T
        )  # (hidden_nodes, input_nodes)

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        _, _, _, final_outputs = self.forward_pass(inputs)
        return final_outputs
