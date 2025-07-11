from collections import Counter
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from secml.array import CArray
from data.mnist_loader import *
from data.cifar_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load and initialize
#tr, val, ts = load_mnist(digits=(4, 6), n_tr=300, n_val=300, n_ts=300)
tr, val, ts = load_data(labels=(0, 8), n_tr=300, n_val=300, n_ts=300)
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


def refined_poison_detection(all_X, all_Y, detected_poisoning_indices, inner_neighbors, threshold=0.9):
    """
    Poisoning olarak isaretlenen her noktanin inner class'larini diger poisoning noktalarinin
    inner class'lariyla karsilastirarak %90 uyusma olanlari gercek poisoning olarak isaretler.
    """
    confirmed_poisoning_indices = []
    
    # Tum poisoning noktalarinin inner komsularini bul
    poisoning_neighbors = {}
    for idx in detected_poisoning_indices:
        neighbors = indices[idx][:inner_neighbors]  # Inner neighbors sec
        poisoning_neighbors[idx] = set(neighbors)
    
    # Poisoning noktalarinin inner class'larini birbiriyle karsilastir
    for idx_i, neighbors_i in poisoning_neighbors.items():
        match_count = 0
        total_comparisons = 0
        
        for idx_j, neighbors_j in poisoning_neighbors.items():
            if idx_i != idx_j:  # Kendiyle karsilastirma yapma
                total_comparisons += 1
                match_ratio = len(neighbors_i.intersection(neighbors_j)) / inner_neighbors
                if match_ratio >= threshold:
                    match_count += 1
        
        # Eğer poisoning noktanin diger poisoning noktalarina gore uyum orani %90 ise, poisoning olarak isaretle
        if total_comparisons > 0 and (match_count / total_comparisons) >= threshold:
            confirmed_poisoning_indices.append(idx_i)
    
    return confirmed_poisoning_indices

# Yeni filtreyi uygula
detected_poisoning_indices_refined = refined_poison_detection(all_X, all_Y, detected_poisoning_indices, inner_neighbors)

# Identify true poisoning indices
true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

# Binary true labels (1 for real poisoning points, 0 for others)
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1
# Yeni etiketi belirle
predicted_labels = np.zeros(len(all_X), dtype=int)
predicted_labels[detected_poisoning_indices_refined] = 1

# Yeni metrikleri hesapla
tp = np.sum((true_labels == 1) & (predicted_labels == 1))
tn = np.sum((true_labels == 0) & (predicted_labels == 0))
fp = np.sum((true_labels == 0) & (predicted_labels == 1))
fn = np.sum((true_labels == 1) & (predicted_labels == 0))

accuracy = (tp + tn) / len(true_labels)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Updated Metrics:\nAccuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")