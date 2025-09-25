import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

hotels = pd.read_csv('datasets/resort_hotel_bookings.csv')
hotels.head()
hotels.info()

hotels['is_canceled'].value_counts()
# not canceled - 28938 (72.2%)
# canceled - 11122 (27.8%)

hotels['reservation_status'].value_counts()
# attended - 28938 (72.2%)
# canceled - 10831 (27%)
# no-show - 291 (0.7%)

cancellations_per_month = hotels.groupby('arrival_date_month')['is_canceled'].mean()
cancellations_per_month.sort_values()

avoid_list = ['reservation_status', 'reservation_status_date']
object_columns = [col for col in hotels.select_dtypes('object').columns if col not in avoid_list]
print(object_columns[:5])

drop_cols = ['country', 'agent', 'company', 'reservation_status_date', 'arrival_date_week_number',
             'arrival_date_day_of_month', 'arrival_date_year']
hotels = hotels.drop(drop_cols, axis=1)
hotels.info()

hotels['meal'] = hotels['meal'].replace({'Undefined':0, 'SC':0, 'BB':1, 'HB':2, 'FB':3})
hotels['meal'].value_counts()

one_hot_columns = ['arrival_date_month', 'distribution_channel', 'reserved_room_type', 'assigned_room_type',
                   'deposit_type', 'customer_type', 'market_segment']

hotels = pd.get_dummies(hotels, columns=one_hot_columns, dtype=int)
hotels.head()

target_cols = ['is_canceled', 'reservation_status']
train_features = [i for i in hotels.columns if i not in target_cols]
train_features

X = torch.tensor(hotels[train_features].values, dtype=torch.float)
y = torch.tensor(hotels['is_canceled'].values, dtype=torch.float).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 18),
    nn.ReLU(),
    nn.Linear(18, 1),
    nn.Sigmoid()
)

loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

num_epochs = 1000
for i in range(num_epochs):
    predictions = model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()

    predicted_labels = (predictions >= .5).int()
    accuracy = accuracy_score(y_train, predicted_labels)

    if (i + 1) % 100 == 0:
        print(f'Epoch[{(i + 1)}/{num_epochs}], BCELoss: {BCELoss.item(): .4f}, Accuracy: {accuracy: .4f}')

model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_predicted_labels = (test_predictions >= .5).int()

print(accuracy_score(y_test, test_predicted_labels))
print(classification_report(y_test, test_predicted_labels))

# Multiclass classification
hotels['reservation_status'] = hotels['reservation_status'].replace({'Check-Out':2, 'Canceled':1, 'No-Show':0})

X = torch.tensor(hotels[train_features].values, dtype=torch.float)
y = torch.tensor(hotels['reservation_status'].values, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(65, 65),
    nn.ReLU(),
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 3)
)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500
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
    multiclass_predictions = model(X_test)
    multiclass_predicted_labels = torch.argmax(multiclass_predictions, dim=1)

print(accuracy_score(y_test, multiclass_predicted_labels))
print(classification_report(y_test, multiclass_predicted_labels))
