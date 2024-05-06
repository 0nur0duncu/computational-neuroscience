import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, max_iterations=1000, error_threshold=0.001):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        
    def activation_function(self, x):
        # Sigmoid aktivasyon fonksiyonu
        return 1 / (1 + np.exp(-x))
    
    def predict(self, inputs):
        # Ağırlıkları ve bias'ı kullanarak tahmin yapma
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(weighted_sum)
    
    def train(self, inputs, targets):
        iteration = 0
        total_error = float('inf')
        
        while iteration < self.max_iterations and total_error > self.error_threshold:
            total_error = 0
            
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = targets[i] - prediction
                total_error += abs(error)
                
                # Ağırlıkları güncelleme
                self.weights += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error
            
            iteration += 1
            
        print(f"Training completed in {iteration} iterations with total error {total_error}")
        print("Weights:", self.weights)
        print("Bias:", self.bias)

# Veri setlerini kullanarak perceptronları eğitme
datasets = [
    {"name": "Test Verisi", "p": np.array([[2, 1, -2, 1], [2, -2, 2, 1]]), "t": np.array([0, 1, 0, 1])},
    {"name": "AND", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 0, 0, 1])},
    {"name": "OR", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 1, 1, 1])},
    {"name": "XOR", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 1, 1, 0])},
    {"name": "Parite 3 Problemi", "p": np.array([[-1, -1, -1, -1, 1, 1, 1, 1], [-1, -1, 1, 1, -1, -1, 1, 1], [-1, 1, -1, 1, -1, 1, -1, 1]]), "t": np.array([-1, 1, 1, -1, 1, -1, -1, 1])}
]

for dataset in datasets:
    print(f"\nTraining Perceptron for {dataset['name']} Dataset:")
    input_size = len(dataset['p'][0])
    perceptron = Perceptron(input_size)
    perceptron.train(dataset['p'], dataset['t'])
