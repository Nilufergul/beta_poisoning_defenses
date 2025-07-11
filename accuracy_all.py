
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from secml.array import CArray
from data.mnist_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack_all import run_attack


tr, val, ts = load_mnist(digits=(4,6), n_tr=100, n_val=400, n_ts=600)


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

#first_poisoning = poisoning_points[0]
#x_poison, y_poison = first_poisoning
#first_proto = x_proto[0]


tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
ts_X = ts.X.tondarray() if isinstance(ts.X, CArray) else ts.X
#x_poison_all = x_poison
x_poison_all = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])

tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y
ts_Y = ts.Y.tondarray() if isinstance(ts.Y, CArray) else ts.Y
#y_poison_all = y_poison
y_poison_all = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])


all_X = np.concatenate([tr_X, ts_X, x_poison_all])
all_Y = np.concatenate([tr_Y, ts_Y, y_poison_all])

# Mean noktasını hesaplamak için class_label1 sınıfındaki veri noktalarını filtreleme
class_label1 = 1 
class_indices1 = np.where(all_Y == class_label1)[0]
class_points1 = all_X[class_indices1]


class_label2 = 0
class_indices2 = np.where(all_Y == class_label2)[0]
class_points2 = all_X[class_indices2]


mean_point = np.mean(class_points1, axis=0)

# Öklidyen mesafeleri hesaplama (class_label2'ye ait noktaların mean_point'e uzaklığı)
distances_to_mean = cdist(class_points2, mean_point.reshape(1, -1), metric='euclidean').flatten()


threshold_distance = 4 # Eşik değerini belirleme
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


print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(all_X)
mean_point_reduced = pca.transform(mean_point.reshape(1, -1))


plt.figure(figsize=(12, 8))
colors = ['blue', 'cyan']
for i, label in enumerate(np.unique(all_Y)):
    indices = np.where(all_Y == label)
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], marker='o', label=f'Class {label}', alpha=0.5, s=50)


#plt.scatter(X_reduced[true_poisoning_indices, 0], X_reduced[true_poisoning_indices, 1], c='red', marker='o', label='Real Poisoned Points', s=15)


plt.scatter(X_reduced[detected_poisoning_indices, 0], X_reduced[detected_poisoning_indices, 1], c='red', marker='x', label='Detected Poisoned Points',alpha=0.7, s=20)

plt.scatter(mean_point_reduced[:, 0], mean_point_reduced[:, 1], c='magenta', marker='X', label='Mean Point', s=100)

plt.legend()
plt.title('Real and Detected Poisoning Points Visualization')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('/Users/nilufergulciftci/desktop/graphs/accuracy_all.png')
plt.show()
