import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# ----------Setting up a Logistic Regression Model----------
# Load the data
health = pd.read_csv("dataR2.csv")
# Split independent and dependent variables
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Fit the model
lr.fit(X, y)
# Print the accuracy of the model
print(lr.score(X, y))


# ----------Sequential Forward Selection----------
# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Sequential forward selection
sfs = SFS(
  lr,
  k_features=3,
  forward=True,
  floating=False,
  scoring='accuracy',
  cv=0
)
# Fit the equential forward selection model
sfs.fit(X, y)


# ----------Evaluating the Result of Sequential Forward Selection----------
# Print the chosen feature names
print(sfs.subsets_[3]['feature_names'])

# Print the accuracy of the model after sequential forward selection
print(sfs.subsets_[3]['avg_score'])

# Plot the model accuracy
plot_sfs(sfs.get_metric_dict())
plt.show()


# ----------Sequential Backward Selection with mlxtend----------
sbs = SFS(lr,
          k_features=3,
          forward=False,
          floating=False,
          scoring='accuracy',
          cv=0)

# Fit sbs to X and y
sbs.fit(X, y)

# Print the chosen feature names
print(sbs.subsets_[3]['feature_names'])

# Print the accuracy of the model after sequential backward selection
print(sbs.subsets_[3]['avg_score'])

# Plot the model accuracy
plot_sfs(sbs.get_metric_dict())
plt.show()


# ----------Sequential Forward and Backward Floating Selection with mlxtend----------
# Sequential forward floating selection
sffs = SFS(lr,
          k_features=3,
          forward=True,
          floating=True,
          scoring='accuracy',
          cv=0)
sffs.fit(X, y)

# Print a tuple with the names of the features chosen by sequential forward floating selection.
print(sffs.subsets_[3]['feature_names'])  # ('Age', 'Glucose', 'Insulin')

# Sequential backward floating selection
sbfs = SFS(lr,
          k_features=3,
          forward=False,
          floating=True,
          scoring='accuracy',
          cv=0)
sbfs.fit(X, y)

# Print a tuple with the names of the features chosen by sequential backward floating selection.
print(sbfs.subsets_[3]['feature_names'])  # ('Age', 'Glucose', 'Resistin')


# ----------Recursive Feature Elimination with scikit-learn----------
# Standardize the data
X = StandardScaler().fit_transform(X)

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

# Recursive feature elimination
rfe = RFE(lr, n_features_to_select=3)
rfe.fit(X, y)


# ----------Evaluating the Result of Recursive Feature Elimination----------
# Create a list of feature names
feature_list = list(X.columns)

# Recursive feature elimination
rfe = RFE(estimator=lr, n_features_to_select=3)
rfe.fit(X, y)

# List of features chosen by recursive feature elimination
rfe_features = [f for f, support in zip(feature_list, rfe.support_) if support]
# Print the accuracy of the model with features chosen by recursive feature elimination
print(rfe.score(X, y))

