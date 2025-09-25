# import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

# print(aaron_judge.columns)
# print(aaron_judge.description.unique())
# print(aaron_judge.type.unique())

aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B':0})
# print(aaron_judge.type)
# print(aaron_judge.plate_x)

aaron_judge = aaron_judge.dropna(subset=['type', 'plate_x', 'plate_z'])
# print(aaron_judge.plate_x)
# plt.scatter(aaron_judge['plate_x'], aaron_judge['plate_z'], c=aaron_judge['type'], cmap=plt.cm.coolwarm, alpha=.25)

training_set, validation_set = train_test_split(aaron_judge, random_state=1)
current_best_gamma = 1
current_best_C = 1
current_best_acc = 0.0
for i in range(1, 101):
  for j in range(1, 101):


    classifier = SVC(kernel='rbf', gamma=i, C=j)

    classifier.fit(training_set[['plate_x', 'plate_z']], training_set[['type']])

# draw_boundary(ax, classifier)
    score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set[['type']])
    # print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set[['type']]), 'i=', i, 'j=', j)

    if score > current_best_acc:
      current_best_acc = score
      current_best_gamma = i
      current_best_C = j

# plt.show()
print(current_best_acc)
print(current_best_gamma)
print(current_best_C)


