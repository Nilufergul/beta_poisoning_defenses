from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from secml.array import CArray
from data.mnist_loader import *
from data.cifar_loader import *
from data.cifar100_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load and initialize
tr, val, ts = load_data100(labels=(46, 99), n_tr=300, n_val=300, n_ts=300)
#tr, val, ts = load_data(labels=(0, 8), n_tr=300, n_val=300, n_ts=300)
clf = LogisticClassifier()
clf.init_fit(tr, {"C": 1})

# Parameters for poisoning
params = {
    "n_proto": 30,
    "lb": 1,
    "y_target": np.array([1]),
    "y_poison": np.array([0]),
    "transform": lambda x: x,
}

# Run the attack
poisoning_points, x_proto = run_attack(beta_poison, "beta_poison_attack", clf, tr, val, ts, params)

# Prepare training data
tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y

all_x_poison = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])
all_y_poison = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])

# Sample a subset of poisoning points
sample_size = int(tr.X.shape[0] * 0.20)
random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
x_poison_all = all_x_poison[random_indices]
y_poison_all = all_y_poison[random_indices]

# Combine original and poisoned data
all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])

# KNN to identify poisoning points
knn = NearestNeighbors(n_neighbors=(len(y_poison_all)*2))
knn.fit(all_X)
distances, indices = knn.kneighbors(all_X)

# Poisoning point count and dynamic neighborhood sizes
poisoning_count = len(y_poison_all)
inner_neighbors = poisoning_count  # Inner group size
outer_neighbors = poisoning_count + int(poisoning_count * 0.2)  # Outer group size with 20% extra

detected_poisoning_indices = []
for i, neighbors in enumerate(indices):
    # Inner and outer group classifications
    inner_classes = all_Y[neighbors[:inner_neighbors]]
    outer_classes = all_Y[neighbors[inner_neighbors:outer_neighbors]]
    most_common_inner_class = np.bincount(inner_classes).argmax()
    most_common_outer_class = np.bincount(outer_classes).argmax()

    # Compare classes between inner and outer groups
    if most_common_inner_class != most_common_outer_class:
        detected_poisoning_indices.append(i)

# Identify true poisoning indices
true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

# Binary true labels (1 for real poisoning points, 0 for others)
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

# Binary predicted labels (1 for detected poisoning points, 0 for others)
predicted_labels = np.zeros(len(all_X), dtype=int)
predicted_labels[detected_poisoning_indices] = 1

tp = np.sum((true_labels == 1) & (predicted_labels == 1))
tn = np.sum((true_labels == 0) & (predicted_labels == 0))
fp = np.sum((true_labels == 0) & (predicted_labels == 1))
fn = np.sum((true_labels == 1) & (predicted_labels == 0))

accuracy = (tp + tn) / len(true_labels)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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
plt.title('Poisoning Points Detected on CIFAR', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=14)
plt.ylabel('PCA Component 2', fontsize=14)
plt.tick_params(axis='both', labelsize=10)

# Save and show plot
plt.tight_layout()

plt.show()