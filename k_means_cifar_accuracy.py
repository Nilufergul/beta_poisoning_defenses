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

# Sabit bir random_state kullanarak rastgeleliği kontrol altına al
random_state = 0
np.random.seed(random_state)


tr, val, ts = load_data(labels=(0, 1), n_tr=100, n_val=400, n_ts=600)


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
ts_X = ts.X.tondarray() if isinstance(ts.X, CArray) else ts.X
x_poison_all = x_poison

tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y
ts_Y = ts.Y.tondarray() if isinstance(ts.Y, CArray) else ts.Y
y_poison_all = y_poison


all_X = np.concatenate([tr_X, ts_X, x_poison_all])
all_Y = np.concatenate([tr_Y, ts_Y, y_poison_all])


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


accuracies = []
precisions = []
recalls = []
f1_scores = []


k_values = range(1, 21)

for k in k_values:
    
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(sorted_distances.reshape(-1, 1))

    
    kmeans_labels = kmeans.labels_

   
    cluster_means = [np.mean(sorted_distances[kmeans_labels == i]) for i in range(k)]

    
    min_mean_cluster = np.argmin(cluster_means)

    
    detected_poisoning_indices_kmeans = class_indices2[sorted_indices[kmeans_labels == min_mean_cluster]]

   
    true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]
   
    true_labels = np.zeros(len(all_X), dtype=int)
    true_labels[true_poisoning_indices] = 1

    predicted_labels_kmeans = np.zeros(len(all_X), dtype=int)
    predicted_labels_kmeans[detected_poisoning_indices_kmeans] = 1

    accuracy_kmeans = accuracy_score(true_labels, predicted_labels_kmeans)
    precision_kmeans = precision_score(true_labels, predicted_labels_kmeans)
    recall_kmeans = recall_score(true_labels, predicted_labels_kmeans)
    f1_kmeans = f1_score(true_labels, predicted_labels_kmeans)

    accuracies.append(accuracy_kmeans)
    precisions.append(precision_kmeans)
    recalls.append(recall_kmeans)
    f1_scores.append(f1_kmeans)


plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', label='Accuracy')
plt.plot(k_values, precisions, marker='o', label='Precision')
plt.plot(k_values, recalls, marker='o', label='Recall')
plt.plot(k_values, f1_scores, marker='o', label='F1-Score')
plt.xlabel('K Değeri')
plt.ylabel('Metrikler')
plt.title('K-Means Sonuçları: K Değerine Göre Performans Metrikleri')
plt.legend()
plt.grid(True)
plt.savefig('zz_cifar/accuracy_cifar_0-1.png')
plt.show()

