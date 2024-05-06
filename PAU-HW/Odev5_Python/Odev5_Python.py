import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# error_threshold=0.001
error_threshold=0
max_epochs=1000
#learning_rate=0.1
learning_rate=1
        
def hardlim(x):
    return 1 if x >= 0 else 0
    
def predict(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return hardlim(weighted_sum)
    
def train(inputs, targets):
    weights = np.random.rand(len(inputs))
    bias = np.random.rand()
    epoch = 0
    total_error = float('inf')
    errors = []
    
    while epoch < max_epochs and total_error > error_threshold:
        total_error = 0
        epoch += 1
        
        for i in range(inputs.shape[1]):
            prediction = predict(inputs[:,i], weights, bias)
            error = targets[i] - prediction
            weights += learning_rate * error * inputs[:,i]
            bias += learning_rate * error
            total_error += abs(error)
        errors.append(total_error)
        print('epoch: %d\t Toplam Hata : %5.3f'%(epoch,total_error))
        
    print(f"Eğitim {epoch} epoch ve {total_error} toplam hata ile tamamlandı.")
    print("Weights:", weights)
    print("Bias:", bias)
    return weights, bias, errors

def plot(name, p, t, weights, bias, errors):
    if name != "Parite 3 Problemi":
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].plot(errors)
        axs[0].set_title(f"{name} hata değişimi")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Toplam Hata")

        axs[1].axhline(0, color='black')
        axs[1].axvline(0, color='black')

        for i in range(p.shape[1]):
            p_i = p[:, i]
            if t[i]:
                axs[1].scatter(p_i[0], p_i[1], s=200, facecolors='black')
            else:
                axs[1].scatter(p_i[0], p_i[1], marker='o', facecolors='none', edgecolors='black', s=200)

        axs[1].set_xlim([-3, 3])
        axs[1].set_ylim([-3, 3])
        x_values = np.linspace(-3, 3, 100)
        y_values = -(weights[0] * x_values + bias) / weights[1]  # Doğru denklemi: w1*x + w2*y + b = 0
        axs[1].plot(x_values, y_values, color='red')
        plt.quiver(0, 0, weights[0], weights[1], angles='xy', scale_units='xy', scale=1, color='blue')

        plt.tight_layout()
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
    weights, bias, errors =train(dataset['p'], dataset['t'])
    plot(dataset['name'],dataset['p'],dataset['t'],weights,bias, errors)
