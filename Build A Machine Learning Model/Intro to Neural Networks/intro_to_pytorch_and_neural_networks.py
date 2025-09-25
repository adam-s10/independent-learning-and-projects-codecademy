import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# -------Intro to tensors-------
# Step 1: Create tensor containing values [2000, 500, 7] with dtype torch.int
apartment_tensor = torch.tensor(np.array([2000, 500, 7]), dtype=torch.int)

# show output
print(apartment_tensor)

# Step 2: Convert dataframe to tensor with dtype torch.float32
# import the dataset using pandas
apartments_df = pd.read_csv("streeteasy.csv")

# select the rent, size, and age columns
apartments_df = apartments_df[["rent", "size_sqft", "building_age_yrs"]]

## YOUR SOLUTION HERE ##
apartments_tensor = torch.tensor(apartments_df.values, dtype=torch.float32)

# show output
print(apartments_tensor)

# -------Linear Regression Review-------
# Step 1: calculate rent for a 500 sqft apartment using equation rent = 3sq_ft + 500
predicted_rent = 3 * 500 + 500
print(predicted_rent)

# Step 2: what is the weight associated with bedrooms for equation rent = 3sz_sqft + 10bedrooms + 250?
bedroom_weight = 10

# Step 3: what is the bias associated with the equation rent = 3sz_sqft + 10bedrooms + 250?
bias = 250

# -------Build a Sequential Neural Network-------
# Step 1: create neural network with 3 input nodes -> 8 hidden nodes using ReLU function -> 1 output node
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
print(model)

# Step 2: recreate the function above but have a second hidden layer with four nodes using sigmoid function
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1)
)
print(model)

# Create model and feedforward all streateasy data
# load pandas DataFrame
apartments_df = pd.read_csv("streeteasy.csv")

# create a numpy array of the numeric columns
apartments_numpy = apartments_df[['size_sqft', 'bedrooms', 'building_age_yrs']].values

# convert to an input tensor
X = torch.tensor(apartments_numpy,dtype=torch.float32)

# preview the first five apartments
print(X[:5])

# Step 3: create the neural network (we are only using random weights and biases at this point)
torch.manual_seed(42)

# define the neural network
model = nn.Sequential(
    nn.Linear(3,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1)
)

## YOUR SOLUTION HERE ##
predicted_rent = model(X)

# show output
print(predicted_rent[:5])
# -------Build a Neural Network Class-------
# set a random seed - do not modify
torch.manual_seed(42)


## create the NN_Regression class
class NN_Regression(nn.Module):
    def __init__(self):
        super(NN_Regression, self).__init__()
        # initialize layers
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 4)
        self.layer4 = nn.Linear(4, 1)

        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # define the forward pass
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x


## create an instance of NN_Regression
model = NN_Regression()

## create an input tensor

apartments_df = pd.read_csv("streeteasy.csv")
numerical_features = ['size_sqft', 'bedrooms', 'building_age_yrs']
X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)

## feedforward to predict rent
predicted_rents = model(X)

## show output
print(predicted_rents[:5])

# set a random seed - do not modify
torch.manual_seed(42)

# Edit to modify num_hidden_nodes at declaration not hard coding it
## create the NN_Regression class
class OneHidden(nn.Module):
    # add a new numHiddenNodes input
    def __init__(self, num_hidden_nodes):
        super(OneHidden, self).__init__()
        # initialize layers
        # 3 input features, variable output features
        self.layer1 = nn.Linear(2, num_hidden_nodes)
        # variable input features, 8 output features
        self.layer2 = nn.Linear(num_hidden_nodes, 1)

        # initialize activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        ## YOUR SOLUTION HERE ##
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


## YOUR SOLUTION HERE ##
model = OneHidden(10)

## do not modify below this comment

# create an input tensor
input_tensor = torch.tensor([3, 4.5], dtype=torch.float32)

# run feedforward
predictions = model(input_tensor)

# show output
print(predictions)

# -------The Loss Function-------
# Step 1: calculate loss manually
## YOUR SOLUTION HERE ##
difference1 = 750 - 1000
difference2 = 1000 - 900
MSE = (difference1 ** 2 + difference2 ** 2) / 2
# MSE stands for mean squared error. It is the difference between actual and predicted for all values passed
# squared and then added together. That total is divided by 2 eg ((actual_y - predicted_y) ** 2 + (actual_z - predicted_z) ** 2) / 2

# show output
print(MSE)

# Step 2: calculate using PyTorch
# define prediction and target tensors
predictions = torch.tensor([-6.9229, -29.8163, -16.0748, -13.2427, -14.1096], dtype=torch.float)
y = torch.tensor([2550, 11500, 3000, 4500, 4795], dtype=torch.float)

## YOUR SOLUTION HERE ##
loss = nn.MSELoss()
MSE = loss(predictions, y)

# show output
print("MSE Loss:", MSE)

# Step 3: take the square root of the MSE to make it more interpretable
## YOUR SOLUTION HERE ##
RMSE = MSE ** .5

# show output
print(RMSE)

# -------The Optimizer-------
# Note: this will be done using Gradient Descent found in the Adam optimizer

# Step 1: Create instance of Adam optimizer
# set a random seed - do not modify
torch.manual_seed(42)

# create neural network
model = nn.Sequential(
    nn.Linear(3,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1)
)

## YOUR SOLUTION HERE ##
optimizer_ = optim.Adam(model.parameters(), lr=.001)

# Step 2: Use the optimizer to improve model
# set a random seed - do not modify
torch.manual_seed(42)

# create neural network
model = nn.Sequential(
    nn.Linear(3,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4,1)
)

# import the data
apartments_df = pd.read_csv("streeteasy.csv")
numerical_features = ['bedrooms', 'bathrooms', 'size_sqft']
X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)
y = torch.tensor(apartments_df['rent'].values,dtype=torch.float)

# forward pass
predictions = model(X)

# define the loss function and compute loss
loss = nn.MSELoss()
MSE = loss(predictions,y)
print('Initial loss is ' + str(MSE))

## YOUR SOLUTION HERE ##
optimizer = optim.Adam(model.parameters(), lr=.001)
# backward pass to determine "downward" direction
MSE.backward()
# apply the optimizer to update weights and biases
optimizer.step()

# feed the data through the updated model and compute the new loss
predictions = model(X)
MSE = loss(predictions,y)
print('After optimization, loss is ' + str(MSE))

# -------Training-------
apartments_df = pd.read_csv("streeteasy.csv")

numerical_features = ['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs',
                      'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher',
                      'has_patio', 'has_gym']

# create tensor of input features
X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)
# create tensor of targets
y = torch.tensor(apartments_df['rent'].values, dtype=torch.float).view(-1,1)

# set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(14, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# MSE loss function + optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## YOUR SOLUTION HERE ##
num_epochs = 50
for epoch in range(num_epochs):
    predictions = model(X)
    MSE = loss(predictions, y)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()  # reset gradients for the next iteration

    ## DO NOT MODIFY ##
    # keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}')

# See the difference when using more epochs for training
## YOUR SOLUTION HERE ##
num_epochs = 500
for epoch in range(num_epochs):
    predictions = model(X)
    MSE = loss(predictions, y)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()  # reset gradients for the next iteration

    ## DO NOT MODIFY ##
    # keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}')

# -------Testing and Evaluation-------
# Step 1: get train and test sets ready using sklearn.model_selection
apartments_df = pd.read_csv("streeteasy.csv")

numerical_features = ['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs',
                      'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher',
                      'has_patio', 'has_gym']

# create tensor of input features
X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)
# create tensor of targets
y = torch.tensor(apartments_df['rent'].values, dtype=torch.float).view(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, train_size=.7, random_state=2)

# Step 2: train model on train and test set

# set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(14, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# MSE loss function + optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## YOUR SOLUTION HERE ##
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    MSE = loss(predictions, y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    ## DO NOT MODIFY ##
    # keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}')

# Step 3: save the model you just trained
## YOUR SOLUTION HERE ##
torch.save(model, 'model.pth')

# Step 4: import a model that has been trained on 20k epochs and evaluate it using the test set from earlier split
## YOUR SOLUTIONS HERE ##
loaded_model = torch.load('model20k.pth')
loaded_model.eval()
with torch.no_grad():
    predictions = loaded_model(X_test)
    test_MSE = loss(predictions, y_test)


# show output
print('Test MSE is ' + str(test_MSE.item()))
print('Test Root MSE is ' + str(test_MSE.item()**(1/2)))

# Step 5: visualize model performance vs actual targets (if the model was perfect, all dots would be on dashed line)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, label='Predictions', alpha=0.5, color='blue')

plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray', linewidth=2,
         label="Actual Rent")
plt.legend()
plt.title('StreetEasy Dataset - Predictions vs Actual Values')
plt.show()
