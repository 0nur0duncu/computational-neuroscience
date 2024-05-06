import numpy as np

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weight initialization
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        
        # Bias initialization
        self.bias_hidden = np.random.rand(self.hidden_size)
        self.bias_output = np.random.rand(self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        # Forward pass
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        return self.output
    
    def backward(self, x, y, output, learning_rate):
        # Backward pass
        self.error = y - output
        
        # Calculate gradients
        delta_output = self.error
        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, delta_output) * learning_rate
        self.bias_output += np.sum(delta_output) * learning_rate
        
        self.weights_input_hidden += np.dot(x.T, delta_hidden) * learning_rate
        self.bias_hidden += np.sum(delta_hidden) * learning_rate
        
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                output = self.forward(X[i])
                self.backward(X[i], y[i], output, learning_rate)
                total_error += np.mean(np.abs(self.error))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {total_error}")
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            output = self.forward(X[i])
            predictions.append(output)
        return predictions

# Örnek giriş ve çıkış verileri
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Perceptron oluşturma ve eğitme
perceptron = Perceptron(input_size=2, hidden_size=3, output_size=1)
perceptron.train(X, y, epochs=1000, learning_rate=0.1)

# Test verileri
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Tahminler
predictions = perceptron.predict(test_data)
print("Predictions:")
for i in range(len(predictions)):
    print(f"{test_data[i]} -> {predictions[i]}")
