import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import torch
from torch import nn
from torch import optim

ev_charging_reports = pd.read_csv('EV charging reports.csv')
print(ev_charging_reports.head())

traffic_reports = pd.read_csv('Local traffic distribution.csv')
print(traffic_reports[:5])

ev_charging_traffic = ev_charging_reports.merge(traffic_reports, left_on='Start_plugin_hour', right_on='Date_from')
print(ev_charging_traffic.info())

drop_cols = ['session_ID', 'Garage_ID', 'User_ID', 'Shared_ID', 'Plugin_category', 'Duration_category',
             'Start_plugin', 'Start_plugin_hour', 'End_plugout', 'End_plugout_hour', 'Date_from', 'Date_to']

ev_charging_traffic = ev_charging_traffic.drop(drop_cols, axis=1)

for col in ev_charging_traffic:
    if ev_charging_traffic[col].dtype == 'object':
        ev_charging_traffic[col] = ev_charging_traffic[col].str.replace(',', '.')

for col in ev_charging_traffic:
    ev_charging_traffic[col] = ev_charging_traffic[col].astype('float')

numerical_features = ['User_private', 'Duration_hours', 'month_plugin_Apr',
       'month_plugin_Aug', 'month_plugin_Dec', 'month_plugin_Feb',
       'month_plugin_Jan', 'month_plugin_Jul', 'month_plugin_Jun',
       'month_plugin_Mar', 'month_plugin_May', 'month_plugin_Nov',
       'month_plugin_Oct', 'month_plugin_Sep', 'weekdays_plugin_Friday',
       'weekdays_plugin_Monday', 'weekdays_plugin_Saturday',
       'weekdays_plugin_Sunday', 'weekdays_plugin_Thursday',
       'weekdays_plugin_Tuesday', 'weekdays_plugin_Wednesday',
       'Kroppan_bru_traffic', 'Moholtlia_traffic', 'Selsbakk_traffic',
       'Moholt_rampe_2_traffic', 'Jonsvannsveien_vest_steinanvegen_traffic']

X = ev_charging_traffic [numerical_features]
y = ev_charging_traffic[['El_kWh']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2, random_state=2)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_model.score(X_test, y_test)


test_mse = mean_squared_error(linear_model.predict(X_test), y_test)
print(test_mse)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1)

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(26, 56),  # use 26 as we use 26 columns as training features
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)

loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=.0007)

num_epochs = 3000
for i in range(num_epochs):
    predictions = model(X_train_tensor)
    MSE = loss(predictions, y_train_tensor)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (i + 1) % 500 == 0:
        print(f'Epoch [{i + 1}/{num_epochs}] MSE Loss: {MSE.item()}')

torch.save(model, 'model.pth')

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = loss(predictions, y_test_tensor)
test_loss.item()

loaded_model = torch.load('model4500.pth')
loaded_model.eval()
with torch.no_grad():
    loaded_predictions = loaded_model(X_test_tensor)
    test_loaded_loss = loss(loaded_predictions, y_test_tensor)
test_loaded_loss.item()




