import pandas as pd
import matplotlib.pyplot as plt

# ----------Implementing a Decision Tree----------
#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

#Loading the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
                 names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])

## 1a. Take a look at the dataset
print(df.head())

## 1b. Setting the target and predictor variables
df['accep'] = ~(df['accep']=='unacc') # True is acceptable, False if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']

## 1c. Examine the new features
print(X.columns)
print(len(X.columns))

print(df.head())

## 2a. Performing the train-test split
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

## 2b.Fitting the decision tree classifier
dt = DecisionTreeClassifier(max_depth=3, ccp_alpha=0.01,criterion='gini')
dt.fit(x_train, y_train)

## 3.Plotting the Tree
plt.figure(figsize=(20,12))
tree.plot_tree(dt, feature_names = x_train.columns, max_depth=5, class_names = ['unacc', 'acc'], label='all', filled=True)
plt.tight_layout()
plt.show()


# ----------Interpreting a Decision Tree----------
## Nothing of note gone through here...just how to interpret the visualization of the tree itself


# ----------Gini Impurity----------
# - Refers to how "pure" the node is, i.e. if the node only contains 1 class in our dataset, it is very pure
# - Gini impurity is highest when p_1 is at 0.5, or a perfect balance of classes in a node


# ----------Information Gain----------
# - Relates to how our tree is performing on information splits, i.e. how much the model is improving by splitting on a feature
# - This calculation is information gain or how much the nodes improved as a result of that split
# - Higher information gain is good and an information gain of 0 means that a split was useless (did not improve our model)
# - We want our gini impurity to be as low as possible and picking features to split on that provide the best information gain is the way to achieve this


# ----------How a Decision Tree is Built (Feature Split)----------
## The usual libraries, loading the dataset and performing the train-test split
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
                 names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep'] == 'unacc')  # 1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:, 0:6])
y = df['accep']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


## Functions to calculate gini impurity and information gain

def gini(data):
    """calculate the Gini Impurity
    """
    data = pd.Series(data)
    return 1 - sum(data.value_counts(normalize=True) ** 2)


def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right branch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)


#### -----------------------------------
## 1. Calculate sample sizes for a split on `persons_2`
left = y_train[(x_train['persons_2'] == 0)]  # get all values in y when corresponding x values are 0
right = y_train[(x_train['persons_2'] == 1)]  # get all values in y when corresponding x values are 1

len_left = len(left)
len_right = len(right)

print('No. of cars with persons_2 == 0:', len_left)
print('No. of cars with persons_2 == 1:', len_right)

## 2. Gini impurity calculations
gi = gini(y_train)

gini_left = gini(left)
gini_right = gini(right)

print('Original gini impurity (without splitting!):', gi)
print('Left split gini impurity:', gini_left)
print('Right split gini impurity:', gini_right)

## 3.Information gain when using feature `persons_2`

info_gain_persons_2 = info_gain(left, right, gi)

print(f'Information gain for persons_2:', info_gain_persons_2)

## 4. Which feature split maximizes information gain?
info_gain_list = []
for i in x_train.columns:
    left = y_train[x_train[i] == 0]
    right = y_train[x_train[i] == 1]
    info_gain_list.append([i, info_gain(left, right, gi)])

info_gain_table = pd.DataFrame(info_gain_list).sort_values(1, ascending=False)
print(f'Greatest impurity gain at:{info_gain_table.iloc[0, :]}')
print(info_gain_table)


# ----------How a Decision Tree is Built (Recursion)----------
def gini(data):
    """calculate the Gini Impurity
    """
    data = pd.Series(data)
    return 1 - sum(data.value_counts(normalize=True) ** 2)


def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right banch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
                 names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep'] == 'unacc')  # 1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:, 0:6])
y = df['accep']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

y_train_sub = y_train[x_train['safety_low'] == 0]
x_train_sub = x_train[x_train['safety_low'] == 0]

gi = gini(y_train_sub)
print(f'Gini impurity at root: {gi}')

info_gain_list = []
for i in x_train.columns:
    left = y_train_sub[x_train_sub[i] == 0]
    right = y_train_sub[x_train_sub[i] == 1]
    info_gain_list.append([i, info_gain(left, right, gi)])

info_gain_table = pd.DataFrame(info_gain_list).sort_values(1, ascending=False)
print(f'Greatest impurity gain at:{info_gain_table.iloc[0, :]}')


# ----------Train and Predict using Scikit-Learn----------
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

## 1. Create a decision tree and print the parameters
dtree = DecisionTreeClassifier()
print(f'Decision Tree parameters: {dtree.get_params()}')

## 2. Fit decision tree on training set and print the depth of the tree

dtree.fit(x_train, y_train)
print(f'Decision tree depth: {dtree.get_depth()}')

## 3. Predict on test data and accuracy of model on test set
y_pred = dtree.predict(x_test)

print(f'Test set accuracy: {dtree.score(x_test, y_pred)}') # or accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {accuracy_score(y_test, y_pred)}')


# ----------Visualizing Decision Trees----------
## Loading the data and setting target and predictor variables
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']

## Train-test split and fitting the tree
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.3)
dtree = DecisionTreeClassifier(max_depth=3)
dtree.fit(x_train, y_train)

## Visualizing the tree
plt.figure(figsize=(27,12))
tree.plot_tree(dtree)
plt.tight_layout()
plt.show()
## Text-based visualization of the tree (View this in the Output terminal!)
print(tree.export_text(dtree, feature_names=x_train.columns.tolist()))
