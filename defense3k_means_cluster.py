import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from secml.array import CArray
from data.mnist_loader import *
from data.cifar_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack
from sklearn.decomposition import PCA

random_state = 0
np.random.seed(random_state)

# Veriyi yükleme ve sınıflandırıcı eğitimi
#tr, val, ts = load_data(labels=(0, 8), n_tr=300, n_val=300, n_ts=300)
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

# Veriyi hazırlama
tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y

all_x_poison = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])
all_y_poison = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])

# Rastgele örnekleme
sample_size = int(tr.X.shape[0] * 0.20)
random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
x_poison_all = all_x_poison[random_indices]
y_poison_all = all_y_poison[random_indices]

all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])

# Veriyi kümelere ayırma
class_label1 = 1  
class_indices1 = np.where(all_Y == class_label1)[0]
class_points1 = all_X[class_indices1]

class_label2 = 0  
class_indices2 = np.where(all_Y == class_label2)[0]
class_points2 = all_X[class_indices2]

mean_point = np.mean(class_points1, axis=0)
distances_to_mean = cdist(class_points2, mean_point.reshape(1, -1), metric='euclidean').flatten()
sorted_indices = np.argsort(distances_to_mean)
sorted_distances = distances_to_mean[sorted_indices].reshape(-1, 1)

# Elbow Method kullanarak en iyi küme sayısını belirleme
k_values = range(3, 10)
sse = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(sorted_distances)
    sse.append(kmeans.inertia_)

# Elbow Method için grafiği çizme
plt.figure(figsize=(10, 5))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method ile Optimal Küme Sayısı')
plt.show()

# Dirsek noktasına göre optimal k değeri seçme
optimal_k = k_values[np.argmin(np.gradient(sse))]
print(f"Optimal Küme Sayısı (Elbow Method'a göre): {optimal_k}")

# Optimal küme sayısına göre KMeans kümeleme işlemi
kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
kmeans.fit(sorted_distances)

kmeans_labels = kmeans.labels_

# En düşük ortalama mesafeye sahip kümeyi seçme
cluster_mean_distances = [np.mean(sorted_distances[kmeans_labels == i]) for i in range(optimal_k)]
min_mean_cluster = np.argmin(cluster_mean_distances)
detected_poisoning_indices = class_indices2[sorted_indices[kmeans_labels == min_mean_cluster]]

# Performans ölçütleri
true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

predicted_labels_kmeans = np.zeros(len(all_X), dtype=int)
predicted_labels_kmeans[detected_poisoning_indices] = 1

tp = np.sum((true_labels == 1) & (predicted_labels_kmeans == 1))
tn = np.sum((true_labels == 0) & (predicted_labels_kmeans == 0))
fp = np.sum((true_labels == 0) & (predicted_labels_kmeans == 1))
fn = np.sum((true_labels == 1) & (predicted_labels_kmeans == 0))

accuracy = (tp + tn) / len(true_labels)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"K-means Precision (k={optimal_k}): {precision:.3f}")
print(f"K-means Recall (k={optimal_k}): {recall:.3f}")
print(f"K-means F1-Score (k={optimal_k}): {f1:.3f}")
print(f"K-means Accuracy (k={optimal_k}): {accuracy:.3f}")

print(accuracy, precision, recall, f1)
# PCA and plot
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(all_X)
unique_classes = np.unique(all_Y)

plt.figure(figsize=(6, 4))
plt.rcParams['legend.fontsize'] = 12

colors = ['magenta', 'blue'] 
# Plot original classes
class_label = unique_classes[0]
class_indices = np.where(all_Y == class_label)[0]
plt.scatter(X_reduced[class_indices, 0], X_reduced[class_indices, 1], 
            label=f'Class {class_label} (Poisoned Class)',marker='o', c=colors[class_label], alpha=0.7, s=75)

class_label = unique_classes[1]
class_indices = np.where(all_Y == class_label)[0]
plt.scatter(X_reduced[class_indices, 0], X_reduced[class_indices, 1], 
            label=f'Class {class_label}',marker='o', c=colors[class_label], alpha=0.8, s=75)

# Highlight detected poisoning points
if len(detected_poisoning_indices) > 0:
    plt.scatter(X_reduced[detected_poisoning_indices, 0], X_reduced[detected_poisoning_indices, 1], 
                c='black', label='Detected Poisoning Points', marker='X', s=50)

# Display metrics on the plot
plt.legend()
plt.title('Poisoning Points Detected on MNIST', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=14)
plt.ylabel('PCA Component 2', fontsize=14)
plt.tick_params(axis='both', labelsize=10)

# Save and show plot
plt.tight_layout()
plt.savefig('/Users/nilufergulciftci/desktop/poisoning/Defense3/defense3_mnist.pdf')
plt.show()
