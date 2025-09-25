from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

points  = labels = None
training_data, validation_data, training_labels, validation_labels = train_test_split(
    points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

"""poly kernel means that it is a 2d representation is transformed into 3d representation/plane"""
classifier = SVC(kernel='poly', degree=2)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

"""linear kernel means that it is a 2d representation remains a 2d representation/plane; C refers to the sensitivity to the training data"""
classifier = SVC(kernel='linear', C=1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

"""rbf or the default kernel converts a 2d representation into an infinite number of dimensions; gamma refers to the sensitivity to the training data"""
classifier = SVC(kernel='rbf', gamma=1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))
