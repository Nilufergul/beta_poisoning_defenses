##bu kodda 
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

# Veri yükleme ve sınıflandırıcı eğitimi
tr, val, ts = load_mnist(digits=(4, 6), n_tr=300, n_val=300, n_ts=300)
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

sample_size = int(tr.X.shape[0] * 0.20)
random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
x_poison_all = all_x_poison[random_indices]
y_poison_all = all_y_poison[random_indices]

all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])

# Calculate 10% of the dataset size for the number of neighbors
n_neighbors = int(len(all_X) * 0.1)
nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(all_X)
distances, indices = nbrs.kneighbors(all_X)

true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

# Binary true labels (1 for real poisoning points, 0 for others)
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

# Initialize lists for metrics
thresholds = np.arange(0, 10, 0.1)
precisions_mnist = []
recalls_mnist = []
f1_scores_mnist = []
accuracies_mnist = []

# Calculate the average distance for each point
average_distances = np.mean(distances, axis=1)

for threshold_distance in thresholds:
    # Identify points where the average distance is below the threshold
    poisoning_indices = np.where(average_distances < threshold_distance)[0]

    # Binary predicted labels (1 for detected poisoning points, 0 for others)
    predicted_labels = np.zeros(len(all_X), dtype=int)
    predicted_labels[poisoning_indices] = 1

    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))

    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    accuracies_mnist.append(accuracy)
    precisions_mnist.append(precision)
    recalls_mnist.append(recall)
    f1_scores_mnist.append(f1)

#best_k_index = np.argmax(f1_scores)
#best_k = thresholds[best_k_index]

#print(f"Best K: {best_k:.3f}")
#print(f"F1-Score: {f1_scores[best_k_index]:.3f}")
#print(f"Precision: {precisions[best_k_index]:.3f}")
#print(f"Recall: {recalls[best_k_index]:.3f}")
#print(f"Accuracy: {accuracies[best_k_index]:.3f}")

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

sample_size = int(tr.X.shape[0] * 0.20)
random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
x_poison_all = all_x_poison[random_indices]
y_poison_all = all_y_poison[random_indices]

all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])

# Calculate 10% of the dataset size for the number of neighbors
n_neighbors = int(len(all_X) * 0.1)
nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(all_X)
distances, indices = nbrs.kneighbors(all_X)

true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

# Binary true labels (1 for real poisoning points, 0 for others)
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

# Initialize lists for metrics
thresholds = np.arange(0, 10, 0.1)
precisions_cifar = []
recalls_cifar = []
f1_scores_cifar = []
accuracies_cifar = []

# Calculate the average distance for each point
average_distances = np.mean(distances, axis=1)

for threshold_distance in thresholds:
    # Identify points where the average distance is below the threshold
    poisoning_indices = np.where(average_distances < threshold_distance)[0]

    # Binary predicted labels (1 for detected poisoning points, 0 for others)
    predicted_labels = np.zeros(len(all_X), dtype=int)
    predicted_labels[poisoning_indices] = 1

    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))

    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    accuracies_cifar.append(accuracy)
    precisions_cifar.append(precision)
    recalls_cifar.append(recall)
    f1_scores_cifar.append(f1)

import matplotlib.ticker as mticker

figure, axis = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

def format_ticks(value, _):
    return f"{value:.3f}"

#plt.rcParams.update({'font.size': 16})
axis[0,0].plot(thresholds, precisions_cifar, label='CIFAR-10', color='blue', linewidth=2)
axis[0,0].plot(thresholds, precisions_mnist, label='MNIST', color='red', linewidth=2)
axis[0,0].set_xlabel('tau', fontsize=14)
axis[0,0].set_ylabel('Precision', fontsize=14)
axis[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axis[0,0].tick_params(axis='both', labelsize=10)
axis[0,0].legend(loc='lower right')

axis[0,1].plot(thresholds, recalls_cifar, label='CIFAR-10', color='blue', linewidth=2)
axis[0,1].plot(thresholds, recalls_mnist, label='MNIST', color='red', linewidth=2)
axis[0,1].set_xlabel('tau', fontsize=14)
axis[0,1].set_ylabel('Recall', fontsize=14)
axis[0,1].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axis[0,1].tick_params(axis='both', labelsize=10)
axis[0,1].legend()

axis[1,0].plot(thresholds, f1_scores_cifar, label='CIFAR-10', color='blue', linewidth=2)
axis[1,0].plot(thresholds, f1_scores_mnist, label='MNIST', color='red', linewidth=2)
axis[1,0].set_xlabel('tau', fontsize=14)
axis[1,0].set_ylabel('F1-Score', fontsize=14)
axis[1,0].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axis[1,0].tick_params(axis='both', labelsize=10)
axis[1,0].legend()

axis[1,1].plot(thresholds, accuracies_cifar, label='CIFAR-10', color='blue', linewidth=2)
axis[1,1].plot(thresholds, accuracies_mnist, label='MNIST', color='red', linewidth=2)
axis[1,1].set_xlabel('tau', fontsize=14)
axis[1,1].set_ylabel('Accuracy', fontsize=14)
axis[1,1].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axis[1,1].tick_params(axis='both', labelsize=9)
axis[1,1].legend()

plt.tight_layout()
plt.savefig('/Users/nilufergulciftci/desktop/poisoning/Defense4/defense1_new2.pdf')
plt.show()