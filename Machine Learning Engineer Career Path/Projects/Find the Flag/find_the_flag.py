import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses',
        'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(df['landmass'].value_counts())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df[(df['landmass'] == 3) | (df['landmass'] == 6)]
print(df_36.head())

#Print the average vales of the predictors for Europe and Oceania
# print(df_36.groupby('landmass')[var].mean())

#Create labels for only Europe and Oceania
labels = (df['landmass'].isin([3,6]))*1

#Print the variable types for the predictors
print(df_36[var].dtypes)

#Create dummy variables for categorical predictors
print(df_36.mainhue.value_counts())
data = pd.get_dummies(df[var])

#Split data into a train and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.4, random_state=1)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []
for depth in depths:
  dtree = DecisionTreeClassifier(random_state=1, max_depth=depth)
  dtree.fit(X_train, y_train)
  acc_depth.append(dtree.score(X_test, y_test))
print(acc_depth)
#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.show()

#Find the largest accuracy and the depth this occurs
max_acc = np.max(acc_depth)
best_depth = depths[np.argmax(acc_depth)]
print(max_acc, best_depth)

#Refit decision tree model with the highest accuracy and plot the decision tree
dtree = DecisionTreeClassifier(random_state=1, max_depth=best_depth)
dtree.fit(X_train, y_train)
plt.figure(figsize=(14,8))
tree.plot_tree(
  dtree,
  feature_names = X_train.columns,
  class_names = ['Europe', 'Oceania'],
  filled=True
)
plt.tight_layout()
plt.show()

#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
acc_pruned = []
ccp_list = np.logspace(-3, 0, num=20)

for ccp in ccp_list:
  dtree = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp)
  dtree.fit(X_train, y_train)
  acc_pruned.append(dtree.score(X_test, y_test))
print(acc_pruned)
#Plot the accuracy vs ccp_alpha
plt.clf()
plt.plot(ccp_list, acc_pruned)
plt.xscale('log')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.show()

#Find the largest accuracy and the ccp value this occurs
max_acc_ccp = np.max(acc_pruned)
best_ccp = ccp_list[np.argmax(acc_pruned)]
print(max_acc_ccp, best_ccp)

#Fit a decision tree model with the values for max_depth and ccp_alpha found above
dtree = DecisionTreeClassifier(random_state=1, ccp_alpha=best_ccp, max_depth=best_depth)
dtree.fit(X_train, y_train)
print(dtree.score(X_test, y_test))

#Plot the final decision tree
plt.figure(figsize=(14,8))
tree.plot_tree(
  dtree,
  feature_names = X_train.columns,
  class_names = ['Europe', 'Oceania'],
  filled=True
)
plt.show()
