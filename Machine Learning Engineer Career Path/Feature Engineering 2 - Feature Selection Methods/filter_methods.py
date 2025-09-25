import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectKBest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

df = pd.DataFrame(data={
    'edu_goal': ['bachelors', 'bachelors', 'bachelors', 'masters', 'masters', 'masters', 'masters', 'phd', 'phd', 'phd'],
    'hours_study': [1, 2, 3, 3, 3, 4, 3, 4, 5, 5],
    'hours_TV': [4, 3, 4, 3, 2, 3, 2, 2, 1, 1],
    'hours_sleep': [10, 10, 8, 8, 6, 6, 8, 8, 10, 10],
    'height_cm': [155, 151, 160, 160, 156, 150, 164, 151, 158, 152],
    'grade_level': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    'exam_score': [71, 72, 78, 79, 85, 86, 92, 93, 99, 100]
})

print(df)


# ----------Variance Threshold----------
X = df.drop(columns=['exam_score'])
print(X)

y = df['exam_score']
print(y)

X_num = X.drop(columns=['edu_goal'])  # For this example, remove any data that is not numerical
print(X_num)

# Create a variance threshold instance where any column that has 0 variance is dropped
selector = VarianceThreshold(threshold=0)  # 0 is default
print(selector.fit_transform(X_num))  # Returns NumPy array

# Get the column names
num_cols = list(X_num.columns[selector.get_support(indices=True)])
print(num_cols)

# Subset X_num to retain only the selected features
X_num = X_num[num_cols]
print(X_num)

# Get the entire features frame including the categorical data
X = X[['edu_goal'] + num_cols]
print(X)


# ----------Pearson's Correlation----------
corr_matrix = X_num.corr(method='pearson')  # 'pearson' is default

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r')
plt.show()

# Loop over bottom diagonal of correlation matrix
for i in range(len(corr_matrix.columns)):
    for j in range(i):

        # Print variables with high correlation
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])

X_y = X_num.copy()
X_y['exam_score'] = y
print(X_y)

corr_matrix = X_y.corr()

# Isolate the column corresponding to `exam_score`
corr_target = corr_matrix[['exam_score']].drop(labels=['exam_score'])
sns.heatmap(corr_target, annot=True, fmt='.3', cmap='RdBu_r')
plt.show()

# Hours study had stronger correlation with exam score so we drop hours TV
X = X.drop(columns=['hours_TV'])
print(X)
# hours sleep and height did not suggest that they were very good predictors, but we will double-check this with another method

# you could also use the f_regression function from scikit-learn as an alternative to pearson's correlation
# the first array is the f-statistic and the second array contains the p value.
# The stronger the correlation: the higher the f value and lower the p value
print(f_regression(X_num, y))


# ----------Mutual Information----------
le = LabelEncoder()

# Create copy of `X` for encoded version
X_enc = X.copy()
X_enc['edu_goal'] = le.fit_transform(X['edu_goal'])
print(X_enc)

# we use mutual_info_regressor as target variable is continuous; if discrete, use mutual_info_classif()
print(mutual_info_regression(X_enc, y, random_state=68))

print(mutual_info_regression(X_enc, y, discrete_features=[0], random_state=68))  # must tell function what variables are discrete

# partial creates a callable partially complete version of the function mutual_info_regression.
score_func = partial(mutual_info_regression, discrete_features=[0], random_state=68)

# Select top 3 features with the most mutual information
selection = SelectKBest(score_func=score_func, k=3)

# here is where the partial function is complete. X_enc and y are passed to mutual_info_regression to return the top 3
# features based on mutual information
print(selection.fit_transform(X_enc, y))


