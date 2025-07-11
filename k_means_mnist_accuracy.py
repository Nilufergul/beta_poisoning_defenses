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


tr, val, ts = load_data(labels=(0,8), n_tr=300, n_val=300, n_ts=300)


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


    tp = np.sum((true_labels == 1) & (predicted_labels_kmeans == 1))
    tn = np.sum((true_labels == 0) & (predicted_labels_kmeans == 0))
    fp = np.sum((true_labels == 0) & (predicted_labels_kmeans == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels_kmeans == 0))

    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

best_k_index = np.argmax(f1_scores)
best_k = k_values[best_k_index]

print(f"Best K: {best_k}")
print(f"F1-Score: {f1_scores[best_k_index]}")
print(f"Precision: {precisions[best_k_index]}")
print(f"Recall: {recalls[best_k_index]}")
print(f"Accuracy: {accuracies[best_k_index]}")  


plt.figure(figsize=(16, 7))
plt.plot(k_values, accuracies, marker='o', label='Accuracy')
plt.plot(k_values, precisions, marker='o', label='Precision')
plt.plot(k_values, recalls, marker='o', label='Recall')
plt.plot(k_values, f1_scores, marker='o', label='F1-Score')
plt.xlabel('K Value', fontsize=14)
plt.ylabel('Metrics', fontsize=14)
plt.title('K-Means Results: Performance Metrics by K Value', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.legend(fontsize=12)
plt.grid(True)
#plt.savefig('/Users/nilufergulciftci/desktop/poisoning/Defense3/k_means_threshold_cifar.png')
plt.show()

