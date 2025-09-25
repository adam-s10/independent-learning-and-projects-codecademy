import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head(), '\n')
print(transactions.info(), '\n')

# How many fraudulent transactions?
print(sum(i for i in transactions['isFraud'] if i == 1), '\n')

# Summary statistics on amount column
print(transactions['amount'].describe(), '\n')

# Create isPayment field
transactions['isPayment'] = [1 if t == 'PAYMENT' or t == 'DEBIT' else 0 for t in transactions['type']]

# Create isMovement field
transactions['isMovement'] = [1 if t == 'CASH_OUT' or t == 'TRANSFER' else 0 for t in transactions['type']]

# Create accountDiff field
account_diff = []
for i in range(len(transactions['type'])):
  diff = abs(
    transactions['oldbalanceOrg'][i] - transactions['oldbalanceDest'][i]
    )
  account_diff.append(diff)
transactions['accountDiff'] = account_diff

# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']
print(features.head())
print(label.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=.3, random_state=0)

# Normalize the features variables
standard_scaler = StandardScaler()
standard_scaler.fit_transform(X_train)
standard_scaler.transform(X_test)

# Fit the model to the training data
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# Score the model on the training data
print(model.score(X_train, y_train))

# Score the model on the test data
print(model.score(X_test, y_test))

# Print the model coefficients
print(model.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([3050.0, 0.0, 1.0, 300.76])

# Combine new transactions into a single array
sample_transactions = np.stack((your_transaction, transaction1, transaction2, transaction3))

# Normalize the new transactions
sample_transactions = standard_scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print(model.predict(sample_transactions))

# Show probabilities on the new transactions
print(model.predict_proba(sample_transactions))