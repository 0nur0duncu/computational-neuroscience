"""

🧑‍💻 Onur Oduncu
🔢 22253503

"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

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
    return np.zeros_like(x)

error_threshold = 0.01

def train(name, a0, t, epochs=10000, learning_rate=0.01,hiddenLayerNeurons = 2):
    global error_threshold
    inputLayerNeurons = a0.shape[1]
    outputLayerNeurons = t.shape[1]
    W1 = np.random.randn(inputLayerNeurons, hiddenLayerNeurons)
    W2 = np.random.randn(hiddenLayerNeurons, outputLayerNeurons)
    b1 = np.random.randn(1, hiddenLayerNeurons)
    b2 = np.random.randn(1, outputLayerNeurons)

    errors = []
    total_error = float('inf')
    epoch = 0
    while epoch < epochs and total_error > error_threshold:
        epoch += 1
        hidden_layer_output = tanh(np.dot(a0, W1) + b1)
        a = tanh(np.dot(hidden_layer_output, W2) + b2)

        error = t - a
        s = error * tanh_derivative(a)
        total_error += abs(np.mean(np.abs(error)))
        errors.append(np.mean(np.abs(error)))

        error_hidden_layer = s.dot(W2.T)
        d_hidden_layer = error_hidden_layer * tanh_derivative(hidden_layer_output)

        W2 += hidden_layer_output.T.dot(s) * learning_rate
        b2 += np.sum(s, axis=0, keepdims=True) * learning_rate
        W1 += a0.T.dot(d_hidden_layer) * learning_rate
        b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    print(f"Output from neural network after {epochs} epochs:")
    print(a)

    if inputLayerNeurons == 2:
        plot_decision_boundary_2d(name, a0, t, W1, b1, W2, b2)
    elif inputLayerNeurons == 3:
        plot_decision_boundary_3d(name, a0, t, W1, b1, W2, b2)
    else:
        print("Input dimension not supported for visualization.")

    plt.show()
    
    plot_errors(name, errors)

def plot_decision_boundary_2d(name, a0, t, W1, b1, W2, b2):
    x_min, x_max = a0[:, 0].min() - 1, a0[:, 0].max() + 1
    y_min, y_max = a0[:, 1].min() - 1, a0[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    hidden_layer_output = tanh(np.dot(grid, W1) + b1)
    Z = tanh(np.dot(hidden_layer_output, W2) + b2)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)

    plt.scatter(a0[:, 0], a0[:, 1], c=t, cmap=plt.cm.binary, s=200)
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    plt.savefig(f'./images/{name}.png')

def plot_decision_boundary_3d(name, a0, t, W1, b1, W2, b2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = a0[:, 0].min() - 1, a0[:, 0].max() + 1
    y_min, y_max = a0[:, 1].min() - 1, a0[:, 1].max() + 1
    z_min, z_max = a0[:, 2].min() - 1, a0[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1),
                             np.arange(z_min, z_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    hidden_layer_output = tanh(np.dot(grid, W1) + b1)
    Z = tanh(np.dot(hidden_layer_output, W2) + b2)
    Z = Z.reshape(xx.shape)

    ax.scatter(xx, yy, zz, c=Z, cmap=plt.cm.binary, alpha=0.6)

    for i in range(t.shape[0]):
        p_i = a0[i, :]
        if t[i]:
            ax.scatter(p_i[0], p_i[1], p_i[2], s=200, facecolors='black')
        else:
            ax.scatter(p_i[0], p_i[1], p_i[2], marker='o', facecolors='none', edgecolors='black', s=200)
            
    plt.savefig(f'./images/{name}.png')

def plot_errors(name, errors):
    plt.plot(errors)
    plt.title('Training Errors')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig(f'./images/{name}_errors.png')
    plt.show()

datasets = [
    {"name": "Test Verisi", "p": np.array([[2, 1, -2, 1], [2, -2, 2, 1]]), "t": np.array([0, 1, 0, 1])},
    {"name": "AND", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 0, 0, 1])},
    {"name": "OR", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 1, 1, 1])},
    {"name": "XOR", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 1, 1, 0])},
    {"name": "Parite 3 Problemi", "p": np.array([[-1, -1, -1, -1, 1, 1, 1, 1], [-1, -1, 1, 1, -1, -1, 1, 1], [-1, 1, -1, 1, -1, 1, -1, 1]]), "t": np.array([-1, 1, 1, -1, 1, -1, -1, 1])}
]

for dataset in datasets:
    print(f"\nTraining Neural Network for {dataset['name']} dataset")
    train(dataset['name'],dataset['p'].T, dataset['t'].reshape((-1, 1)))
