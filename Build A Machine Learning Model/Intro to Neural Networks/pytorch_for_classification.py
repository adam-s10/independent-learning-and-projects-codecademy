import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# -----------Encodings-----------
# create the sample dataframe
df = pd.DataFrame({'Student_ID':[1,2,3,4,5],
                   'Letter_Grade':['A','C','F','B','D'],
                   'Outcome':['Passed','Passed','Failed','Passed','Failed']})

# Label encode Letter_Grade and Outcome columns
df['Letter_Grade'] = df['Letter_Grade'].replace(
    {'A':4,
    'B':3,
    'C':2,
    'D':1,
    'F':0})

df['Outcome'] = df['Outcome'].replace(
    {'Passed':1,
     'Failed':0})

print(df.head())

# One-Hot Encoding
# create the sample dataframe
df = pd.DataFrame({'Student_ID':[1,2,3,4,5],
                   'High_School_Type':['State','Private','Other','State', 'State']})

# One-hot encode High_School_Type column
df = pd.get_dummies(
    df,
    columns=['High_School_Type'],
    dtype=int)

print(df.head())

# label encode
# create the dataframe df
df = pd.DataFrame({'Student_ID':[1,2,3,4,5],
                   'Additional_Work':['Yes','Yes','No','Yes','No'],
                   'Regular_Artistic_or_Sports':['No','Yes','Yes','No','No'],
                   'Has_Partner':['No','Yes','No','Yes','Yes']})

## YOUR SOLUTION HERE ##
replace_dict = {'Yes': 1, 'No': 0}

df['Additional_Work'] = df['Additional_Work'].replace(replace_dict)
df['Regular_Artistic_or_Sports'] = df['Regular_Artistic_or_Sports'].replace(replace_dict)
df['Has_Partner'] = df['Has_Partner'].replace(replace_dict)


# show encoded output
print(df.head())

# One-hot encode
# create the dataframe df
df = pd.DataFrame({'Student_ID':[1,2,3,4,5],
                   'MT1_Preparation':['Alone', 'Alone', 'With Friends', 'Alone', 'Not Applicable']})

## YOUR SOLUTION HERE ##
df = pd.get_dummies(
    df,
    columns=['MT1_Preparation'],
    dtype=int
)

# show encoded output
print(df.head())

# -----------Sigmoids and Thresholds-----------
# import Python's math module to access the exponential function math.exp
import math

# define sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Define inputs for a student who studies but does not take notes
studies = 0.1
notes = 0.1

# Multiply each input by the corresponding weight
weighted_studies = 2.5 * studies
weighted_notes = 5.0 * notes

# Calculate weighted sum - this should produce 2.5 as in the narrative
weighted_sum = weighted_studies + weighted_notes
print("Weighted Sum:", weighted_sum)

# Apply the sigmoid activation function (run the setup cell if you haven't yet)
predicted_probability = sigmoid(weighted_sum)

# Determine a prediction using a threshold of .5
threshold = 0.75
classification = int(predicted_probability >= threshold)

# Print probability and classification
print("Probability:", predicted_probability)
print("Classification:", classification)

## YOUR SOLUTION HERE ##

# Define inputs
studies = 1.0
notes = 0.0
gpa = 3.0

# Multiply each input by the corresponding weight
weighted_studies = studies * -5
weighted_notes = notes * -4
weighted_last_gpa = gpa * 2.2

# Calculate weighted sum
weighted_sum = weighted_studies + weighted_notes + weighted_last_gpa

# Apply the sigmoid activation function (run the setup cell if you haven't yet)
predicted_probability = sigmoid(weighted_sum)

# Determine a prediction using a threshold of .5
threshold = 0.5
classification = int(predicted_probability >= threshold)

# Print probability and classification
print("Probability:", predicted_probability)
print("Classification:", classification)

# re-calculating the probability from the prior checkpoint for ease
predicted_probability = sigmoid(1.6)

## YOUR SOLUTION HERE ##
threshold = .85
classification = int(predicted_probability >= threshold)

# Print probability and classification - do not modify
print("Probability:", predicted_probability)
print("Classification:", classification)

# Use pytorch to build neural network with Sigmoid funtion
## YOUR SOLUTION HERE ##
model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)

# -----------Training-----------
# Step 1: import data and split into train and test set
# Load the dataset
df = pd.read_csv("student_performances_encoded.csv")

# Remove columns from training:
# - Student_ID since it is just a unique row identifier
# - Letter_Grade since that contains info that directly determines the target column
# - Outcome since that is the target column

remove_cols = ['Student_ID', 'Letter_Grade', 'Outcome']
train_features = [x for x in df.columns if x not in remove_cols]

# Create tensor of input features
X = torch.tensor(df[train_features].values, dtype=torch.float)
# Create tensor of targets
y = torch.tensor(df['Outcome'].values, dtype=torch.float).view(-1,1)

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80, # use 80% of the data for training
                                                    test_size=0.20, # use 20% of the data for testing
                                                    random_state=42) # set a random state

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Step 2: Create a model and train using train data

# Set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(55, 110), # 55 is the number of input features in X_train
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    nn.Linear(55, 1), # one output node for binary classification
    nn.Sigmoid() # sigmoid activation to output probabilities
)

## YOUR SOLUTION HERE ##

# Import accuracy_score function from sklearn.metrics
from sklearn.metrics import accuracy_score

# initialize the BCE loss function
loss = nn.BCELoss()

# initialize SGD optimizer, with a learning rate of .001
optimizer = optim.SGD(model.parameters(), lr=.001)

# set the number of epochs to 300
num_epochs = 300

for epoch in range(num_epochs):
    ## Add forward pass here, keep the variable name predictions ##
    predictions = model(X_train)

    ## Compute BCELoss loss here ##
    BCELoss = loss(predictions, y_train)

    ## Compute gradients here ##
    BCELoss.backward()

    ## Update weights and biases here ##
    optimizer.step()

    ## Reset the gradients for the next iteration here ##
    optimizer.zero_grad()

    ## DO NOT MODIFY ##
    # keep track of the accuracy and loss during training
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions >= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')


# Step 3: adjust learning rate of the model to determine if it improves
## YOUR SOLUTION HERE ##
# Set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(55, 110), # 55 is the number of input features in X_train
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    nn.Linear(55, 1), # one output node for binary classification
    nn.Sigmoid() # sigmoid activation to output probabilities
)

## YOUR SOLUTION HERE ##

# Import accuracy_score function from sklearn.metrics
from sklearn.metrics import accuracy_score

# initialize the BCE loss function
loss = nn.BCELoss()

# initialize SGD optimizer, with a learning rate of .01
optimizer = optim.SGD(model.parameters(), lr=.01)

# set the number of epochs to 300
num_epochs = 300

for epoch in range(num_epochs):
    ## Add forward pass here, keep the variable name predictions ##
    predictions = model(X_train)

    ## Compute BCELoss loss here ##
    BCELoss = loss(predictions, y_train)

    ## Compute gradients here ##
    BCELoss.backward()

    ## Update weights and biases here ##
    optimizer.step()

    ## Reset the gradients for the next iteration here ##
    optimizer.zero_grad()

    ## DO NOT MODIFY ##
    # keep track of the accuracy and loss during training
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions >= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Step 4: increase epochs using new learning rate to see if it improves
## YOUR SOLUTION HERE ##
# Set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(55, 110), # 55 is the number of input features in X_train
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    nn.Linear(55, 1), # one output node for binary classification
    nn.Sigmoid() # sigmoid activation to output probabilities
)

## YOUR SOLUTION HERE ##

# Import accuracy_score function from sklearn.metrics
from sklearn.metrics import accuracy_score

# initialize the BCE loss function
loss = nn.BCELoss()

# initialize SGD optimizer, with a learning rate of .01
optimizer = optim.SGD(model.parameters(), lr=.01)

# set the number of epochs to 1000
num_epochs = 1000

for epoch in range(num_epochs):
    ## Add forward pass here, keep the variable name predictions ##
    predictions = model(X_train)

    ## Compute BCELoss loss here ##
    BCELoss = loss(predictions, y_train)

    ## Compute gradients here ##
    BCELoss.backward()

    ## Update weights and biases here ##
    optimizer.step()

    ## Reset the gradients for the next iteration here ##
    optimizer.zero_grad()

    ## DO NOT MODIFY ##
    # keep track of the accuracy and loss during training
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions >= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')


# -----------Evaluation-----------
# Step 1: train the neural network
# Set a random seed
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(55, 110),
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    nn.Linear(55, 1),
    nn.Sigmoid()
)

# BCE loss function + SGD optimizer
loss = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # keep track of the accuracy and loss during training
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions >= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Step 2: generate predictions
model.eval()
with torch.no_grad():
    ## YOUR SOLUTION HERE ##
    test_predictions = model(X_test)
    test_predicted_labels = (test_predictions >= .5).int()

# show output - do not remove or modify
print(test_predicted_labels)

# Step 3: Evaluate model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## YOUR SOLUTION HERE ##
test_accuracy = accuracy_score(y_test, test_predicted_labels)
test_precision = precision_score(y_test, test_predicted_labels)
test_recall = recall_score(y_test, test_predicted_labels)
test_f1 = f1_score(y_test, test_predicted_labels)

# show output - do not remove or modify
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1)

# -----------Multiclass models-----------
# Suppose we wanted to predict whether a student:
# - always takes notes
# - almost always takes notes
# - sometimes takes notes
# - never takes notes

# Set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(5, 110),
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    ## YOUR SOLUTION HERE ##
# instead of outputting 1, we output 4 as we are trying to predict which of 4 classes is more likely
    nn.Linear(55, 4)

)

# -----------Multiclass softmax-----------
# Softmax works by converting all the probabilities outputted to add up to 1. Eg, we could be trying to predict which of
# 3 classes our features belong to and get the following probabilities: [.9, .8, .4]. Softmax will convert this list to
# probabilities that add up to 1.

# First it calculates normalization factor. It does so by applying the exponential function to each value in the list
# and summing them. We then apply the exponential function to the individual probabilities and divide by the normalization factor.

normalization_factor = (np.exp(.9) + np.exp(.8) + np.exp(.4))

softmax_9 = np.exp(.9) / normalization_factor

## YOUR SOLUTION HERE ##

softmax_8 = np.exp(.8) / normalization_factor

softmax_4 = np.exp(.4) / normalization_factor

# show output
print(np.round(softmax_9,2))
print(np.round(softmax_8,2))

# -----------Multiclass argmax-----------
# We use argmax to make the final classification. Argmax examines all the probabilities and makes the classification
# based off the highest one. Eg [0.1320, 0.0160, 0.9614, 0.9919] would identify index 3 as the highest probability and
# classify this input as 3

# we have data for 5 students and our model has outputted 4 probabilities that correspond to:
# - 0: always takes notes
# - 1: almost always takes notes
# - 2: sometimes takes notes
# - 3: never takes notes
raw_output = torch.tensor([
    [0.1320, 0.0160, 0.9614, 0.9919],
    [0.7180, 0.7303, 0.6234, 0.1197],
    [0.8757, 0.2045, 0.1977, 0.3845],
    [0.8934, 0.5677, 0.1377, 0.6420],
    [0.4017, 0.8363, 0.1119, 0.6557]
],
    dtype = torch.float)

## YOUR SOLUTION HERE ##
largest_output = 0.9919
largest_output_index = 3
predicted_label = 3

# show output - do not modify
print("For the first student, the largest output is",largest_output)
print("This corresponds to index",largest_output_index)
print("The predicted label is",predicted_label)

# instead of inputting manually, use argmax to make the prediction for all students
## YOUR SOLUTION HERE ##
argmax_output = torch.argmax(raw_output, dim=1)

# show output - do not modify
print(argmax_output)


# -----------Multiclass train and evaluate-----------
# Step 1: load data:
df = pd.read_csv("student_performances_encoded.csv")

# Create Performance_Outcome target column {0: Below Average, 1: Average, 2: Above Average}
df['Performance_Outcome'] = df['Letter_Grade'].replace({0:0, 1:0,
                                                        2:1, 3:1,
                                                        4:2, 5:2})

# Preview the dataset
print(df.head())

# Step 2: create features and target variables
# Creating list of training features
remove_cols = ['Student_ID', 'Letter_Grade', 'Outcome', 'Performance_Outcome']
train_features = [x for x in df.columns if x not in remove_cols]

## YOUR SOLUTION HERE ##

# Create float tensor of input features
X = torch.tensor(df[train_features].values, dtype=torch.float)
# Create long tensor of multiclass targets
y = torch.tensor(df['Performance_Outcome'].values, dtype=torch.long)

# show output - do not modify
print(X)

# Step 3: split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8, # use 80% of the data for training
                                                    test_size=0.2, # use 20% of the data for testing
                                                    random_state=42) # set a random state

# Step 3: train the multiclass neural network
# set a random seed - do not modify
torch.manual_seed(42)

# define a model
model = nn.Sequential(
    nn.Linear(55, 240),
    nn.ReLU(),
    nn.Linear(240, 110),
    nn.ReLU(),
    nn.Linear(110, 3)
)

## YOUR SOLUTION HERE ##
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    CELoss = loss(predictions, y_train)
    CELoss.backward()
    optimizer.step()
    optimizer.zero_grad()

    ## DO NOT MODIFY ##
    # keep track of the loss and accuracy during training
    if (epoch + 1) % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], CELoss: {CELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Step 4: evaluate model on test set
from sklearn.metrics import accuracy_score, classification_report

model.eval()
with torch.no_grad():
    ## YOUR SOLUTION HERE ##
    predictions = model(X_test)
    predicted_labels = torch.argmax(predictions, dim=1)
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)

# show output - do not modify
print(f'Accuracy: {accuracy.item():.4f}')
print(report)

# -----------Review-----------
import torch
from torch import nn
from torch import optim

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('student_performances_encoded.csv')
df.head()

df.columns

df.info()

features = ['Scholarship_Type', 'Weekly_Study_Hours', 'Attendance_Seminars_Conferences', 'Attendance_Class',
            'Last_Cumulative_GPA', 'Expected_GPA']
classes = df['Letter_Grade']
df['Letter_Grade'].value_counts()

X = torch.tensor(df[features].values, dtype=torch.float)
y = torch.tensor(classes.values, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2, random_state=42)

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(6, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Sigmoid(),
    nn.Linear(128, 64),
    nn.Sigmoid(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 5)
)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.085)

num_epochs = 40000
for i in range(num_epochs):
    predictions = model(X_train)
    CELoss = loss(predictions, y_train)
    CELoss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (i + 1) % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch[{(i + 1)}/{num_epochs}], CELoss: {CELoss.item():.4f}, Accuracy: {accuracy:.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_labels = torch.argmax(predictions, dim=1)
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)

print(accuracy)
print(report)
# 40000 epochs + lr=0.05 = .41 accuracy


