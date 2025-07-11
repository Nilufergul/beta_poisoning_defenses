import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from secml.array import CArray
from data.mnist_loader import *
from data.cifar_loader import *

from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack

random_state = 0
np.random.seed(random_state)


tr, val, ts = load_data(labels=(0, 8), n_tr=300, n_val=300, n_ts=300)


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

tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y

all_x_poison = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])
all_y_poison = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])

# tr size'ın %20'si kadar bir örnekleme boyutu belirliyoruz
sample_size = int(tr.X.shape[0] * 0.20)

# Rastgele sample_size kadar noktayı seçiyoruz
random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
x_poison_all = all_x_poison[random_indices]
y_poison_all = all_y_poison[random_indices]

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


sorted_indices = np.argsort(distances_to_mean)
sorted_distances = distances_to_mean[sorted_indices]


kmeans = KMeans(n_clusters=10, random_state=random_state)
kmeans.fit(sorted_distances.reshape(-1, 1))


kmeans_labels = kmeans.labels_


cluster_mean_distances = [np.mean(sorted_distances[kmeans_labels == i]) for i in range(10)]


min_mean_cluster = np.argmin(cluster_mean_distances)


detected_poisoning_indices_kmeans = class_indices2[sorted_indices[kmeans_labels == min_mean_cluster]]


true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]


true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1


predicted_labels_kmeans = np.zeros(len(all_X), dtype=int)
predicted_labels_kmeans[detected_poisoning_indices_kmeans] = 1


precision_kmeans = precision_score(true_labels, predicted_labels_kmeans)
recall_kmeans = recall_score(true_labels, predicted_labels_kmeans)
f1_kmeans = f1_score(true_labels, predicted_labels_kmeans)
accuracy_kmeans = accuracy_score(true_labels, predicted_labels_kmeans)

print(f"K-means Precision (k=10): {precision_kmeans:.2f}")
print(f"K-means Recall (k=10): {recall_kmeans:.2f}")
print(f"K-means F1-Score (k=10): {f1_kmeans:.2f}")
print(f"K-means Accuracy (k=10): {accuracy_kmeans:.2f}")


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(all_X)
mean_point_reduced = pca.transform(mean_point.reshape(1, -1))

plt.figure(figsize=(12, 7))


for i, label in enumerate(np.unique(all_Y)):
    indices = np.where(all_Y == label)
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], marker='o', label=f'Class {label}', alpha=0.5, s=50)


plt.scatter(X_reduced[detected_poisoning_indices_kmeans, 0], X_reduced[detected_poisoning_indices_kmeans, 1], 
            c='red', marker='X', label='K-means Detected Poisoned Points', s=20)


plt.scatter(mean_point_reduced[:, 0], mean_point_reduced[:, 1], c='magenta', marker='X', label='Mean Point', s=100)

plt.legend()
plt.title('K-means (10 Cluster) ile Tespit Edilen Zehirlenmiş Noktalar')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('/Users/nilufergulciftci/desktop/graphs/k_means_cifar_0_8.png')
plt.show()