from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()


def multi_model(test_size, random_state):
    (training_data, validation_data,
     training_labels, validation_labels) = train_test_split(breast_cancer_data.data, breast_cancer_data.target,
                                                            test_size=test_size, random_state=random_state)

    accuracies = []
    k_list = range(1, 101)
    for k in k_list:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(training_data, training_labels)
        accuracies.append(classifier.score(validation_data, validation_labels))

    # plt.plot(k_list, accuracies)
    # plt.xlabel('k')
    # plt.ylabel('Validation Accuracy')
    # plt.title('Breast Cancer Classifier Accuracy')
    # plt.show()

    best_to_worst_accuracies = sorted(
        list(enumerate(accuracies)),
        key=lambda x: x[1],
        reverse=True
    )
    best_k = best_to_worst_accuracies[0]
    # TODO: use an odd number k to avoid a tie during prediction
    return best_k


final_accuracies = []
increase_test_size = [.2, .25, .3, .35, .40, .45, .5]
for i in increase_test_size:
    for j in range(1, 101):
        final_accuracies.append([i, j, multi_model(i, j)])

print(final_accuracies)
top_10 = sorted(list(final_accuracies), key=lambda x: x[2][1], reverse=True)[:10]
print(top_10)
# [[0.2, 42, (10, 0.9824561403508771)], [0.2, 60, (20, 0.9824561403508771)], [0.2, 71, (21, 0.9824561403508771)],
# [0.2, 75, (22, 0.9824561403508771)], [0.3, 42, (9, 0.9824561403508771)], [0.4, 42, (9, 0.9824561403508771)],
# [0.35, 42, (9, 0.98)], [0.25, 42, (10, 0.9790209790209791)], [0.25, 52, (2, 0.9790209790209791)],
# [0.25, 60, (44, 0.9790209790209791)]]

# best would be [0.2, 71, (21, .....)] using an odd k means no ties can occur and as all accuracies up to that point are
# equal it is an easy decision
