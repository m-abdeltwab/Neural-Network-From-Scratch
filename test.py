import pandas as pd

# data = pd.read_csv("./data/mnist/mnist_train.csv").to_numpy()
data = pd.read_csv("./data/mnist_train.csv").to_numpy()

df = pd.DataFrame(data[:10000, :])
df.to_csv("train.csv", header=False, index=False)

pd.read_csv("./train.csv").to_numpy()
