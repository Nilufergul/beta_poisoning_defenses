import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from secml.array import CArray
from data.mnist_loader import *
from data.cifar_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack


tr, val, ts = load_data(labels=(1, 9), n_tr=600, n_val=600, n_ts=600)

clf = LogisticClassifier()
clf.init_fit(tr, {"C": 1})


params = {
    "n_proto": 15,
    "lb": 1,
    "y_target": np.array([1]),
    "y_poison": np.array([0]),
    "transform": lambda x: x,
}


poisoning_points, x_proto = run_attack(beta_poison, "beta_poison_attack", clf, tr, val, ts, params)

first_poisoning = poisoning_points[0]
x_poison, y_poison = first_poisoning
first_proto = x_proto[0]

tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
x_poison_all = x_poison
#x_poison_all = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])

tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y
y_poison_all = y_poison
#y_poison_all = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])


all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])


class_label1 = 1
class_indices1 = np.where(all_Y == class_label1)[0]
class_points1 = all_X[class_indices1]


class_label2 = 0 
class_indices2 = np.where(all_Y == class_label2)[0]
class_points2 = all_X[class_indices2]


mean_point = np.mean(class_points1, axis=0)


distances_to_mean = cdist(class_points2, mean_point.reshape(1, -1), metric='euclidean').flatten()


sorted_distances_indices = np.argsort(distances_to_mean)
sorted_distances = distances_to_mean[sorted_distances_indices]


Q1 = np.percentile(sorted_distances, 25)  # 1. çeyrek (25. yüzdelik dilim)
Q3 = np.percentile(sorted_distances, 75)  # 3. çeyrek (75. yüzdelik dilim)
IQR = Q3 - Q1

threshold_distance = Q1 - 1.5 * IQR

detected_poisoning_indices = class_indices2[distances_to_mean < threshold_distance]


true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

# Binary true labels (1 for real poisoning points, 0 for others)
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

# Binary predicted labels (1 for detected poisoning points, 0 for others)
predicted_labels = np.zeros(len(all_X), dtype=int)
predicted_labels[detected_poisoning_indices] = 1

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

print("Mesafeler (küçükten büyüğe):")
print(sorted_distances)


print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1-Score: {f1:.5f}")
print(f"Accuracy: {accuracy:.5f}")

from sklearn.metrics import confusion_matrix

# Confusion Matrix hesapla
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)

# Remove detected poisoning points and evaluate performance
remaining_indices = np.setdiff1d(np.arange(len(all_X)), detected_poisoning_indices)

# Remaining data after removing detected poisoning points
remaining_X = all_X[remaining_indices]
remaining_Y = all_Y[remaining_indices]

# Remaining labels
remaining_true_labels = true_labels[remaining_indices]
remaining_predicted_labels = predicted_labels[remaining_indices]

# Calculate performance on remaining data
remaining_precision = precision_score(remaining_true_labels, remaining_predicted_labels)
remaining_recall = recall_score(remaining_true_labels, remaining_predicted_labels)
remaining_f1 = f1_score(remaining_true_labels, remaining_predicted_labels)
remaining_accuracy = accuracy_score(remaining_true_labels, remaining_predicted_labels)

# Print remaining performance metrics
print(f"Remaining Data Precision: {remaining_precision:.5f}")
print(f"Remaining Data Recall: {remaining_recall:.5f}")
print(f"Remaining Data F1-Score: {remaining_f1:.5f}")
print(f"Remaining Data Accuracy: {remaining_accuracy:.5f}")