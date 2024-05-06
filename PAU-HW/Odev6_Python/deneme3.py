import numpy as np 
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def predict(inputs, weights_hidden, bias_hidden, weights_output, bias_output):
    hidden_layer_activation = np.dot(inputs, weights_hidden)
    hidden_layer_activation += bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_output)
    output_layer_activation += bias_output
    return sigmoid(output_layer_activation)

def train(inputs, expected_output, epochs=10000, learning_rate=0.01):
    inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

    hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
    hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
    output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
    output_bias = np.random.uniform(size=(1, outputLayerNeurons))

    for _ in range(epochs):
        # Forward Propagation
        hidden_layer_activation = np.dot(inputs, hidden_weights)
        hidden_layer_activation += hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, output_weights)
        output_layer_activation += output_bias
        predicted_output = sigmoid(output_layer_activation)

        # Backpropagation
        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating Weights and Biases
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return hidden_weights, hidden_bias, output_weights, output_bias

def plot_predicted_output(inputs, predicted_output):
    plt.scatter(inputs[:,0], inputs[:,1], c=predicted_output.flatten(), cmap='viridis')
    plt.title('Output from Neural Network')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.colorbar(label='Predicted Output')
    plt.show()

# Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

# Training the neural network
hidden_weights, hidden_bias, output_weights, output_bias = train(inputs, expected_output)

# Predicting the output
predicted_output = predict(inputs, hidden_weights, hidden_bias, output_weights, output_bias)
print("\nOutput from neural network after 10,000 epochs:")
print(predicted_output.flatten())

# Plotting the output
plot_predicted_output(inputs, predicted_output)
