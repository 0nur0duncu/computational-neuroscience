"""

Onur Oduncu
22253503

"""


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

def softmax_derivative(x):
    s = softmax(x).reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def hardlim(x):
    return np.where(x >= 0, 1, 0)

def hardlim_derivative(x):
    return 0

def train(a0,t,epochs=10000, learning_rate = 0.01, inputLayerNeurons = 2, hiddenLayerNeurons = 2, outputLayerNeurons = 1):
  W1 = np.random.randn(inputLayerNeurons,hiddenLayerNeurons)
  W2 = np.random.randn(hiddenLayerNeurons,outputLayerNeurons)

  b1 =np.random.randn(1,hiddenLayerNeurons)
  b2 = np.random.randn(1,outputLayerNeurons)

  errors = []

  for _ in range(epochs):
    hidden_layer_output = tanh(np.dot(a0,W1) + b1)
    a = linear(np.dot(hidden_layer_output,W2) + b2)

    error = t - a
    s = error * tanh_derivative(a)
    errors.append(error)

    error_hidden_layer = s.dot(W2.T)
    d_hidden_layer = error_hidden_layer * linear_derivative(hidden_layer_output)

    #Updating Weights and Biases
    W2 += hidden_layer_output.T.dot(s) * learning_rate
    b2 += np.sum(s,axis=0,keepdims=True) * learning_rate
    W1 += a0.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer,axis=0,keepdims=True) * learning_rate

  print(f"\nOutput from neural network after {epochs} epochs: ",end='')
  print(*a)
  if inputLayerNeurons > 2:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_min, x_max = a0[:, 0].min() - 1, a0[:, 0].max() + 1
    y_min, y_max = a0[:, 1].min() - 1, a0[:, 1].max() + 1
    z_min, z_max = a0[:, 2].min() - 1, a0[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1),
                            np.arange(z_min, z_max, 0.1))

    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = sigmoid(np.dot(sigmoid(np.dot(points, W1) + b1), W2) + b2)
    Z = Z.reshape(xx.shape)

    ax.scatter(xx, yy, zz, c=Z, cmap='coolwarm', alpha=0.6)
    for i in range(t.shape[0]):
        p_i = a0[i, :]
        if t[i]:
            ax.scatter(p_i[0], p_i[1], p_i[2], s=200, facecolors='black')
        else:
            ax.scatter(p_i[0], p_i[1], p_i[2], marker='o', facecolors='none', edgecolors='black', s=200)
    
  else:
    x_min, x_max = a0[:, 0].min() - 1, a0[:, 0].max() + 1
    y_min, y_max = a0[:, 1].min() - 1, a0[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    Z = sigmoid(np.dot(sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    for i in range(t.shape[0]):
        p_i = a0[i, :]
        if t[i]:
            plt.scatter(p_i[0], p_i[1], s=200, facecolors='black')
        else:
            plt.scatter(p_i[0], p_i[1], marker='o', facecolors='none', edgecolors='black', s=200)

  plt.show()


datasets = [
    {"name": "Test Verisi", "p": np.array([[2, 1, -2, 1], [2, -2, 2, 1]]), "t": np.array([0, 1, 0, 1])},
    {"name": "AND", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 0, 0, 1])},
    {"name": "OR", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 1, 1, 1])},
    {"name": "XOR", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 1, 1, 0])},
    {"name": "Parite 3 Problemi", "p": np.array([[-1, -1, -1, -1, 1, 1, 1, 1], [-1, -1, 1, 1, -1, -1, 1, 1], [-1, 1, -1, 1, -1, 1, -1, 1]]), "t": np.array([-1, 1, 1, -1, 1, -1, -1, 1])}
]

for dataset in datasets:
    print(f"\nTraining Perceptron for {dataset['name']} Dataset:")
    train(dataset['p'].T, dataset['t'].reshape((-1, 1)),inputLayerNeurons=len(dataset['p']))