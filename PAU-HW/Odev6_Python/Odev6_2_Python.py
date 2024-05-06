import numpy as np

# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid'in türevi
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR mantıksal işlemi için eğitim verileri ve etiketler
training_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
training_outputs = np.array([[0],[1],[1],[0]])

# Ağırlıkların ve bias'ın rastgele başlatılması
np.random.seed(1)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Ağırlıkların rastgele başlatılması
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))

# Bias'ın rastgele başlatılması
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

# Öğrenme oranı
learning_rate = 0.01
epochs = 10000

# Eğitim
for epoch in range(epochs):
    # İleri besleme
    hidden_layer_activation = np.dot(training_inputs, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    # Hata hesaplama
    error = training_outputs - predicted_output
    
    # Geri yayılım
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Ağırlıkların güncellenmesi
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += training_inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Eğitilmiş ağı test etme
hidden_layer_activation = np.dot(training_inputs, hidden_weights) + hidden_bias
hidden_layer_output = sigmoid(hidden_layer_activation)

output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
predicted_output = sigmoid(output_layer_activation)

print("Eğitim verilerine göre tahminler:")
print(predicted_output)