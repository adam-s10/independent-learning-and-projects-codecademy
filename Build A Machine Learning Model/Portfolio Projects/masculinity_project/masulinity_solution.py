import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

masculinity = pd.read_csv('masculinity.csv')

print(masculinity.head())
print(masculinity.columns)

# Cluster responses on question 7 in the pdf
print(masculinity['q0007_0001'].value_counts())
q_seven_cols = ['q0007_0001', 'q0007_0002', 'q0007_0003', 'q0007_0004', 'q0007_0005', 'q0007_0006', 'q0007_0007',
                'q0007_0008', 'q0007_0009', 'q0007_0010', 'q0007_0011']
sub_set = masculinity[q_seven_cols]
# drop rows with missing answers
for col in q_seven_cols:
    sub_set = sub_set.drop(sub_set[sub_set[col] == 'No answer'].index)
print(sub_set['q0007_0001'].value_counts())
# Map str to int while maintaining order
map_dict = {'Often':4, 'Sometimes':3, 'Rarely':2, 'Never, but open to it':1, 'Never, and not open to it':0}

for col in q_seven_cols:
    sub_set[col] = sub_set[col].map(map_dict)
print(sub_set['q0007_0001'].value_counts())

plt.scatter(sub_set['q0007_0001'], sub_set['q0007_0002'], alpha=.1)
plt.show()

cols_of_interest = ['q0007_0001', 'q0007_0002', 'q0007_0003', 'q0007_0004', 'q0007_0005', 'q0007_0008', 'q0007_0009']
model = KMeans(n_clusters=2)
model.fit(sub_set[cols_of_interest])
print(model.cluster_centers_)
