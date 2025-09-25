from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# ---------Confusion Matrix---------

actual = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
predicted = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(actual)):
  if actual[i] == 1 and predicted[i] == 1:
    true_positives += 1
  elif actual[i] == 1 and predicted[i] == 0:
    false_negatives += 1
  elif actual[i] == 0 and predicted[i] == 1:
    false_positives += 1
  else:
    true_negatives += 1

print(true_positives)
print(true_negatives)
print(false_positives)
print(false_negatives)

conf_matrix = confusion_matrix(actual, predicted)
print(conf_matrix)

# ---------Accuracy---------
accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
print(accuracy)

# ---------Recall---------
recall = true_positives/(true_positives + false_negatives)
print(recall)

# ---------Precision---------
precision = true_positives/(true_positives + false_positives)
print(precision)
# ---------F-1 Score---------
f_1 = (2 * precision * recall)/(precision + recall)
print(f_1)

# ---------Metrics Using sklearn---------

print(accuracy_score(actual, predicted))
print(recall_score(actual, predicted))
print(precision_score(actual, predicted))
print(f1_score(actual, predicted))