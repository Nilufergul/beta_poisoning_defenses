import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from secml.array import CArray
from data.mnist_loader import *
from data.cifar_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Veri yükleme ve sınıflandırıcı eğitimi (mevcut kod ile aynı)
tr, val, ts = load_data(labels=(0, 8), n_tr=300, n_val=300, n_ts=300)
clf = LogisticClassifier()
clf.init_fit(tr, {"C": 1})

params = {
    "n_proto": 30,
    "lb": 1,
    "y_target": np.array([0]),
    "y_poison": np.array([1]),
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

# Farklı k-komşular için metrikleri depolamak için listeler
neighbor_values = range(1, 65, 5)  # Neighbor aralığını 1'den 10'a kadar genişlettik
precisions_neighbors = []
recalls_neighbors = []
f1_scores_neighbors = []
accuracies_neighbors = []

for n_neighbors in neighbor_values:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(all_X)
    distances, indices = nbrs.kneighbors(all_X)

    true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

    # Binary true labels (1 for real poisoning points, 0 for others)
    true_labels = np.zeros(len(all_X), dtype=int)
    true_labels[true_poisoning_indices] = 1

    # Eşik değer olarak en büyük mesafeyi alıyoruz
    threshold_distance = 4.0

    # Eşik mesafeye göre zehirlenmiş noktaların tespiti
    poisoning_indices = np.where(distances[:, -1] < threshold_distance)[0]

    # Binary predicted labels (1 for detected poisoning points, 0 for others)
    predicted_labels = np.zeros(len(all_X), dtype=int)
    predicted_labels[poisoning_indices] = 1

    # Metrikleri hesapla
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Metrikleri listelere ekle
    precisions_neighbors.append(precision)
    recalls_neighbors.append(recall)
    f1_scores_neighbors.append(f1)
    accuracies_neighbors.append(accuracy)


import matplotlib.ticker as mticker

# Y eksenini formatlamak için bir işlev
def format_ticks(value, _):
    return f"{value:.3f}"

# Çoklu metrik grafikleri
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Precision grafiği
axs[0, 0].plot(neighbor_values, precisions_neighbors, label='MNIST', color='blue')
axs[0, 0].set_title('Precision')
axs[0, 0].set_xlabel('Number of Neighbors (k)')
axs[0, 0].set_ylabel('Precision')
axs[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axs[0, 0].legend()

# Recall grafiği
axs[0, 1].plot(neighbor_values, recalls_neighbors, label='MNIST', color='blue')
axs[0, 1].set_title('Recall')
axs[0, 1].set_xlabel('Number of Neighbors (k)')
axs[0, 1].set_ylabel('Recall')
axs[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axs[0, 1].legend()

# F1-Score grafiği
axs[1, 0].plot(neighbor_values, f1_scores_neighbors, label='MNIST', color='blue')
axs[1, 0].set_title('F1-Score')
axs[1, 0].set_xlabel('Number of Neighbors (k)')
axs[1, 0].set_ylabel('F1-Score')
axs[1, 0].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axs[1, 0].legend()

# Accuracy grafiği
axs[1, 1].plot(neighbor_values, accuracies_neighbors, label='MNIST', color='blue')
axs[1, 1].set_title('Accuracy')
axs[1, 1].set_xlabel('Number of Neighbors (k)')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axs[1, 1].legend()

fig.suptitle('Results of Discriminator-based defense with different Number of Neighbors (k)', fontsize=16)
plt.tight_layout()
plt.savefig('/Users/nilufergulciftci/desktop/graphs/neighbor_knn_cifar.png')
plt.show()