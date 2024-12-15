import numpy as np
import pandas as pd


def load_data():

    # Load MNIST data
    train_data = pd.read_csv("./data/train.csv").to_numpy()
    test_data = pd.read_csv("./data/test.csv").to_numpy()

    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    return x_train, y_train, x_test, y_test


def normalize(images):
    # Normalize to range [0.01, 1.0]
    images = (images / 255.0) * 0.99 + 0.01
    return images


def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((len(labels), num_classes)) + 0.01
    for i, label in enumerate(labels):
        encoded[i, label] = 0.99
    return encoded
