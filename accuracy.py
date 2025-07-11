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


tr, val, ts = load_mnist(digits=(4, 6), n_tr=300, n_val=300, n_ts=300)

clf = LogisticClassifier()
clf.init_fit(tr, {"C": 1})


params = {
    "n_proto": 30,
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

threshold_distance = Q1 - 1.0 * IQR

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

real_poisoning_distances_to_mean = cdist(x_poison_all, mean_point.reshape(1, -1), metric='euclidean').flatten()

# Uzaklıkları sıralı hale getir
sorted_poisoning_distances = np.sort(real_poisoning_distances_to_mean)

print("Poisoning Points Mesafeler (küçükten büyüğe):")
print(sorted_poisoning_distances)

print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1-Score: {f1:.5f}")
print(f"Accuracy: {accuracy:.5f}")

from sklearn.metrics import confusion_matrix

# Confusion Matrix hesapla
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(all_X)
mean_point_reduced = pca.transform(mean_point.reshape(1, -1))


plt.figure(figsize=(12, 10))


plt.subplot(2, 1, 1)


for i, label in enumerate(np.unique(all_Y)):
    indices = np.where(all_Y == label)
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], marker='o', label=f'Class {label}', alpha=0.5, s=50)


if len(detected_poisoning_indices) > 0:
    plt.scatter(X_reduced[detected_poisoning_indices, 0], X_reduced[detected_poisoning_indices, 1], 
                c='red', marker='X', label='Detected Poisoned Points', s=20)


plt.scatter(mean_point_reduced[:, 0], mean_point_reduced[:, 1], c='magenta', marker='X', label='Mean Point', s=100)


plt.legend()
plt.title('Detected Poisoning Points, Classes and Mean Point')
plt.xlabel('Component 1')
plt.ylabel('Component 2')


plt.subplot(2, 1, 2)


plt.text(0.5, 0.5, f'Threshold: {threshold_distance}\n'
                   f'Precision: {precision}\n'
                   f'Recall: {recall}\n'
                   f'F1-Score: {f1}\n'
                   f'Accuracy: {accuracy}', 
         horizontalalignment='center', verticalalignment='center', fontsize=15)
plt.axis('off')  

plt.tight_layout()
plt.savefig('z_get_threshold/cifar_threshold_0-8.png')
plt.show()


# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(all_X)
# mean_point_reduced = pca.transform(mean_point.reshape(1, -1))


# plt.figure(figsize=(12, 8))
# colors = ['blue', 'cyan', 'green']
# for i, label in enumerate(np.unique(all_Y)):
#     indices = np.where(all_Y == label)
#     plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], marker='o', label=f'Class {label}', alpha=0.5, s=50)


# #plt.scatter(X_reduced[true_poisoning_indices, 0], X_reduced[true_poisoning_indices, 1], c='pink', marker='o', label='Real Poisoned Points', s=100)


# plt.scatter(X_reduced[detected_poisoning_indices, 0], X_reduced[detected_poisoning_indices, 1], c='red', marker='X', label='Detected Poisoned Points', s=20)

# plt.scatter(mean_point_reduced[:, 0], mean_point_reduced[:, 1], c='magenta', marker='X', label='Mean Point', s=100)

# plt.legend()
# plt.title('Real and Detected Poisoning Points Visualization')
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.savefig('z_defense_by_distance/accuracy.png')
# plt.show()
