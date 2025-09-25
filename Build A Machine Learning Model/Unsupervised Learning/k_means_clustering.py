import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from copy import deepcopy


iris = datasets.load_iris()
# Store iris.data
samples = iris.data
target = iris.target

# ----------- Using Scikit-Learn -----------
# Use KMeans() to create a model that finds 3 clusters
model = KMeans(n_clusters=3)  # by not providing keyword argument init='random' this is k-means++
# Use .fit() to fit the model to samples
model.fit(samples)
# Use .predict() to determine the labels of samples
labels = model.predict(samples)
# Print the labels
print(labels)

# Store the new Iris measurements
new_samples = np.array(
    [[5.7, 4.4, 1.5, 0.4],
     [6.5, 3. , 5.5, 0.4],
     [5.8, 2.7, 5.1, 1.9]]
)
print(new_samples)

# Predict labels for the new_samples
print(model.predict(new_samples))

# Make a scatter plot of x and y and using labels to define the colors
x = samples[:,0]
y = samples[:,1]

plt.scatter(x, y, c=labels, alpha=.5)
plt.show()

# use cross tabulation to evaluate performance
species = np.chararray(target.shape, itemsize=150)

for i in range(len(samples)):
  if target[i] == 0:
    species[i] = 'setosa'
  elif target[i] == 1:
    species[i] = 'veriscolor'
  elif target[i] == 2:
    species[i] = 'virginica'

df = pd.DataFrame({'labels':labels, 'species':species})
print(df)

ct = pd.crosstab(df['labels'], df['species'])
print(ct)

# determine the best value for k
num_clusters = [i for i in range(1, 9)]
inertias = []

for i in num_clusters:
  model = KMeans(n_clusters=i)
  model.fit(samples)
  inertias.append(model.inertia_)

plt.plot(num_clusters, inertias, '-o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# ----------- Explore the data and make the algorithm myself -----------


print(iris.data)
print(iris.target)
print(iris.DESCR)

# visualize data before k means


# Create x and y
x = samples[:,0]
y = samples[:,1]
# Plot x and y
plt.scatter(x, y, alpha=.5)
# Show the plot
plt.show()

# implement k-random centroids for initial clusters
# Number of clusters
k = 3
# Create x coordinates of k random centroids
centroids_x = np.random.uniform(min(x), max(x), k)
# Create y coordinates of k random centroids
centroids_y = np.random.uniform(min(y), max(y), k)
# Create centroids array
centroids = np.array(list(zip(centroids_x, centroids_y)))
print(centroids)

# Make a scatter plot of x, y
plt.scatter(y, x)
# Make a scatter plot of the centroids
plt.scatter(centroids_y, centroids_x)
# Display plot
plt.show()

# implement assign values to nearest centroids
sepal_length_width = np.array(list(zip(x, y)))

# Step 2: Assign samples to nearest centroid

# Distance formula
def distance(a, b):
  sum_ = 0
  for j in range(len(a)):
    sum_ += (a[j] - b[j]) ** 2
  return sum_ ** .5

# To store the value of centroids when it updates
centroids_old = np.zeros(centroids.shape)
# Cluster labels for each point (either 0, 1, or 2)
labels = np.zeros(len(samples))
# Distances to each centroid
distances = np.zeros(k)
# Initialize error:
error = np.zeros(3)
for i in range(len(error)):
  error[i] = distance(centroids[i], centroids_old[i])

# Repeat Steps 2 and 3 until convergence:
while error.all() != 0:

    # Assign to the closest centroid
    for i in range(len(samples)):
      distances[0] = distance(sepal_length_width[i], centroids[0])
      distances[1] = distance(sepal_length_width[i], centroids[1])
      distances[2] = distance(sepal_length_width[i], centroids[2])
      cluster = np.argmin(distances)
      labels[i] = cluster

    # Print labels
    print(labels)

    # Step 3: Update centroids
    centroids_old = deepcopy(centroids)

    for i in range(k):
      points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
      centroids[i] = np.mean(points, axis=0)

    print(centroids_old)
    print(centroids)

    for i in range(len(error)):
        error[i] = distance(centroids[i], centroids_old[i])

colors = ['r', 'g', 'b']

for i in range(k):
    points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=.5)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()
