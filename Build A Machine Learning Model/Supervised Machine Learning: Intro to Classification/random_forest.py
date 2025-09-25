from cars import training_points, training_labels, testing_points, testing_labels
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=2000, random_state=0)

classifier.fit(training_points, training_labels)

print(classifier.score(testing_points, testing_labels))