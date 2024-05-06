import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# np.random.seed(42)

# Datasets
datasets = [
    {"name": "Test Verisi", "p": np.array([[2, 1, -2, 1], [2, -2, 2, 1]]), "t": np.array([0, 1, 0, 1])},
    {"name": "AND", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 0, 0, 1])},
    {"name": "OR", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 1, 1, 1])},
    # {"name": "XOR", "p": np.array([[0, 0, 1, 1], [0, 1, 0, 1]]), "t": np.array([0, 1, 1, 0])},
    {"name": "Parite 3 Problemi", "p": np.array([[-1, -1, -1, -1, 1, 1, 1, 1], [-1, -1, 1, 1, -1, -1, 1, 1], [-1, 1, -1, 1, -1, 1, -1, 1]]), "t": np.array([-1, 1, 1, -1, 1, -1, -1, 1])}
]

def plot(name,p,t,weight,bias):
  plt.title(name)
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  
  w,h = p.shape
  
  if w>=2 and h>=2:
    for i in range(h):
      p_i = p[:,i]
      if t[i]:
        plt.scatter(p_i[0],p_i[1],s=200,facecolors='black')
      else:
        plt.scatter(p_i[0],p_i[1],marker='o', facecolors='none',edgecolors='black',s=200)
    
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    x_values = np.linspace(-3, 3, 100)
    y_values = -(weight[0][0] * x_values + bias) / weight[0][1]  # Doğru denklemi: w1*x + w2*y + b = 0
    plt.plot(x_values, y_values, color='red')
    plt.title(name)
    plt.show()

def hardlim(x):
  return 1 if x >= 0 else 0

def Perceptron(name,p,t):
  w = np.random.rand(1,2)
  b = np.random.rand()
  
  nop = p.shape[1]
  epoch = 0
  loop = 1
  E = []
  
  while loop:
    epoch += 1
    te = 0
    for k in range(nop):
      a = hardlim(w.dot(p[:,k]) + b)
      e = t[k] - a
      w = w + e * p[:,k].transpose()
      b = b + e
      te = te + abs(e)
    if not te:
      loop = 0
    E.append(te)
    print('epoch: %d\t Toplam Hata : %5.3f'%(epoch,te))
  
  plt.plot(E)
  plt.title(f"{name} hatası")
  plt.show()
  
  plot(name,p,t,w,b)
  

for dataset in datasets:
  Perceptron(dataset["name"],dataset["p"],dataset["t"])
