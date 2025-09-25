import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.DESCR)
print(digits.data)
print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()

print(digits.target[100])

model = KMeans(n_clusters=10)
model.fit(digits.data)

figure = plt.figure(figsize=(8, 3))
figure.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = figure.add_subplot(2, 5, i + 1)
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.11,4.27,5.34,2.29,0.00,0.00,0.00,0.69,7.62,7.47,7.17,6.71,0.00,0.00,0.00,0.00,2.06,0.39,5.08,6.86,0.00,0.00,0.00,0.00,0.31,4.89,7.62,4.68,0.00,0.00,0.46,4.12,7.32,7.62,4.81,0.84,0.15,0.00,6.46,7.62,7.62,7.62,7.62,7.62,7.62,1.88,3.03,3.81,2.75,2.29,2.29,2.67,3.66,0.69],
[0.00,0.00,1.22,1.52,1.45,0.05,0.00,0.00,0.00,3.49,7.62,7.62,7.62,5.26,3.12,0.13,0.84,7.62,5.19,4.17,5.41,6.79,7.62,5.54,2.29,7.62,1.45,0.00,0.00,0.00,3.56,7.62,2.29,7.62,0.84,0.00,0.00,0.00,2.90,7.62,1.76,7.62,3.87,0.92,0.00,2.59,7.32,5.87,0.15,6.64,7.62,7.63,7.01,7.62,5.95,0.76,0.00,0.30,1.38,3.74,4.57,3.18,0.23,0.00],
[0.00,0.00,0.00,0.15,2.21,3.36,0.00,0.00,0.00,0.00,1.88,6.68,7.62,7.22,0.00,0.00,0.00,0.00,6.10,6.97,2.90,0.53,0.00,0.00,0.00,0.69,7.62,7.01,4.55,0.28,0.00,0.00,0.00,1.14,7.47,5.73,7.62,4.40,0.00,0.00,0.00,4.17,3.20,3.58,7.22,5.34,0.00,0.00,0.00,6.51,7.62,7.62,6.55,2.09,0.00,0.00,0.00,0.15,0.76,0.61,0.00,0.00,0.00,0.00],
[0.00,2.21,2.21,0.00,3.66,5.04,0.00,0.00,0.00,5.49,5.34,0.00,4.57,6.10,0.00,0.00,0.00,6.94,4.88,0.00,5.65,5.95,0.00,0.00,0.91,7.62,5.19,3.05,6.94,5.72,1.68,0.05,1.53,7.62,7.62,7.62,7.62,7.62,7.62,2.12,0.28,2.97,0.74,1.07,7.62,3.43,2.21,0.28,0.00,0.00,0.00,1.76,7.62,1.68,0.00,0.00,0.00,0.00,0.00,2.97,7.62,1.00,0.00,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
