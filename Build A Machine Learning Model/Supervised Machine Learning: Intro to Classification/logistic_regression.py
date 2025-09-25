import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

codecademyU = pd.read_csv('codecademyU_2.csv')


def fitting_model_in_sklearn():
    # Separate out X and y
    X = codecademyU[['hours_studied', 'practice_test']]
    y = codecademyU.passed_exam

    # Transform X
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 27)

    # Create and fit the logistic regression model here:

    cc_lr = LogisticRegression()
    cc_lr.fit(X_train, y_train)

    # Print the intercept and coefficients here:
    print(cc_lr.coef_)
    print(cc_lr.intercept_)


def predictions_in_sklearn():
    # Separate out X and y
    X = codecademyU[['hours_studied', 'practice_test']]
    y = codecademyU.passed_exam

    # Transform X
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

    # Create and fit the logistic regression model here:
    cc_lr = LogisticRegression()
    cc_lr.fit(X_train, y_train)

    # Print out the predicted outcomes for the test data
    y_predictions = cc_lr.predict(X_test)
    print(y_predictions)
    # Print out the predicted probabilities for the test data
    print(cc_lr.predict_proba(X_test))
    # Print out the true outcomes for the test data
    print(y_test)
    print(y_predictions)


def classification_thresholding():
    # Pick an alternative threshold here:
    alternative_threshold = .58

    # Separate out X and y
    X = codecademyU[['hours_studied', 'practice_test']]
    y = codecademyU.passed_exam

    # Transform X
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

    # Create and fit the logistic regression model here:
    cc_lr = LogisticRegression()
    cc_lr.fit(X_train, y_train)

    # Print out the predicted outcomes for the test data
    print(cc_lr.predict(X_test))

    # Print out the predicted probabilities for the test data
    print(cc_lr.predict_proba(X_test)[:, 1])

    # Print out the true outcomes for the test data
    print(y_test)


def confusion_matrix_eg():
    # Separate out X and y
    X = codecademyU[['hours_studied', 'practice_test']]
    y = codecademyU.passed_exam

    # Transform X
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

    # Create and fit the logistic regression model here:
    cc_lr = LogisticRegression()
    cc_lr.fit(X_train, y_train)

    # Save and print the predicted outcomes
    y_pred = cc_lr.predict(X_test)
    print('predicted classes: ', y_pred)

    # Print out the true outcomes for the test data
    print('true classes: ', y_test)

    # Print out the confusion matrix here
    print(confusion_matrix(y_test, y_pred))


def accuracy_recall_precision_f1():
    # Separate out X and y
    X = codecademyU[['hours_studied', 'practice_test']]
    y = codecademyU.passed_exam

    # Transform X
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=51)

    # Create and fit the logistic regression model here:
    cc_lr = LogisticRegression()
    cc_lr.fit(X_train, y_train)

    # Save and print the predicted outcomes
    y_pred = cc_lr.predict(X_test)
    print('predicted classes: ', y_pred)

    # Print out the true outcomes for the test data
    print('true classes: ', y_test)

    # Print out the confusion matrix
    print('confusion matrix: ')
    print(confusion_matrix(y_test, y_pred))

    # Print accuracy here:
    print(accuracy_score(y_test, y_pred))

    # Print F1 score here:
    print(f1_score(y_test, y_pred))


def review():
    cancer_data = load_breast_cancer()
    model = LogisticRegression(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, train_size=.2,
                                                        random_state=1)
    model.fit(X_train, y_train)
    y_predictions = model.predict(X_test)
    print(confusion_matrix(y_test, y_predictions))
    print(accuracy_score(y_test, y_predictions))  # 0.918
    print(f1_score(y_test, y_predictions))  # 0.937
    # print(y_test)
    # print(y_predictions)
    # print(model.predict_proba(X_test))
    '''
    Find another dataset for binary classification from Kaggle or take a look at sklearn‘s breast cancer dataset. 
    Use sklearn to build your own Logistic Regression model on the data and make some predictions. Which features are 
    most important in the model you build?
    '''


review()
