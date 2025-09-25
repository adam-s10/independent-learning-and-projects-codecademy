import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv('income.csv', header=0, delimiter=', ', engine='python')
FOREST = 'Forest:'
TREE = 'Tree:'

print('Data Types: \n', income_data.dtypes)

print(income_data.iloc[0], '\n')

# Look at what values we are working with for each column
print(income_data['income'].value_counts(), '\n')
print(income_data['workclass'].value_counts(), '\n')  # use 'Private' as 0 else 1
print(income_data['education'].value_counts(), '\n')  # might need to be excluded but also good candidate for ranking (0-14)
print(income_data['marital-status'].value_counts(), '\n') # maybe use 'Married-civ-spouse' as 0 else 1
print(income_data['occupation'].value_counts(), '\n')  # numbers very close together and large range of industries; best to avoid
print(income_data['relationship'].value_counts(), '\n')  # again not hugely insightful
print(income_data['race'].value_counts(), '\n')  # use white as 0 else 1
print(income_data['sex'].value_counts(), '\n')  # Male 0, Female 1
print(income_data['native-country'].value_counts(), '\n')


labels = income_data[['income']]
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

# Create classifiers
tree_ = tree.DecisionTreeClassifier(random_state=1)
forest = RandomForestClassifier(random_state=1)

forest.fit(train_data, train_labels)
tree_.fit(train_data, train_labels)
print('\n')
print(FOREST)
print('age, capital-gain, capital-loss, hours-per-week:', forest.score(test_data, test_labels))
print(TREE)
print('age, capital-gain, capital-loss, hours-per-week:', tree_.score(test_data, test_labels))

# Convert 'sex' column strings to integers to add it to data
income_data['sex-int'] = income_data['sex'].apply(lambda row: 0 if row == 'Male' else 1)
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int']]

# create new train and test sets with 'sex-int' included to compare to previous results
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

forest.fit(train_data, train_labels)
tree_.fit(train_data, train_labels)
print('\n')
print(FOREST)
print('age, capital-gain, capital-loss, hours-per-week, sex-int:', forest.score(test_data, test_labels))
print(TREE)
print('age, capital-gain, capital-loss, hours-per-week, sex-int:', tree_.score(test_data, test_labels))

# convert 'native-country' column strings to integers to add to data
print('\n')
print(income_data['native-country'].value_counts())  # look at the occurrences of each value in native-country column
income_data['country-int'] = income_data['native-country'].apply(lambda row: 0 if row == 'United-States' else 1)
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

forest.fit(train_data, train_labels)
tree_.fit(train_data, train_labels)
print(forest.feature_importances_)
print(FOREST)
print('age, capital-gain, capital-loss, hours-per-week, sex-int, country-int:', forest.score(test_data, test_labels))
print(TREE)
print('age, capital-gain, capital-loss, hours-per-week, sex-int, country-int:', tree_.score(test_data, test_labels))

# convert 'race' column strings to integers to add to data
print('\n')
print(income_data['race'].value_counts())  # look at the occurrences of each value in race column
income_data['race-int'] = income_data['race'].apply(lambda row: 0 if row == 'White' else 1)
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int', 'race-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

forest.fit(train_data, train_labels)
tree_.fit(train_data, train_labels)
print(forest.feature_importances_)
print(FOREST)
print('age, capital-gain, capital-loss, hours-per-week, sex-int, country-int, rate-int:',
      forest.score(test_data, test_labels))
print(TREE)
print('age, capital-gain, capital-loss, hours-per-week, sex-int, country-int, race-int:',
      tree_.score(test_data, test_labels))

# convert 'workclass' column strings to integers to add to data
print('\n')
print(income_data['workclass'].value_counts())  # look at the occurrences of each value in workclass column
income_data['workclass-int'] = income_data['workclass'].apply(lambda row: 0 if row == 'Private' else 1)
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int', 'race-int',
                    'workclass-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

forest.fit(train_data, train_labels)
tree_.fit(train_data, train_labels)
print(forest.feature_importances_)
print(FOREST)
print('age, capital-gain, capital-loss, hours-per-week, sex-int, country-int, rate-int, workclass-int:',
      forest.score(test_data, test_labels))
print(TREE)
print('age, capital-gain, capital-loss, hours-per-week, sex-int, country-int, race-int, workclass-int:',
      tree_.score(test_data, test_labels))

# convert 'marital-status' column strings to integers to add to data
print('\n')
print(income_data['marital-status'].value_counts())  # look at the occurrences of each value in marital-status column
income_data['marital-int'] = income_data['marital-status'].apply(lambda row: 0 if row == 'Married-civ-spouse' else 1)
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int', 'race-int',
                    'marital-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

forest.fit(train_data, train_labels)
tree_.fit(train_data, train_labels)
print(forest.feature_importances_)
print(FOREST)
print('age, capital-gain, capital-loss, hours-per-week, sex-int, country-int, rate-int, workclass-int, marital-int:',
      forest.score(test_data, test_labels))
print(TREE)
print('age, capital-gain, capital-loss, hours-per-week, sex-int, country-int, race-int, workclass-int, marital-int:',
      tree_.score(test_data, test_labels))

# TODO convert education to ints...
# df['example_column'] = df['example_column'].map({'A':1, 'B':2})  # use this to make multiple transformations

