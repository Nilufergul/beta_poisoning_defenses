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
from scipy.spatial.distance import pdist

# Load and initialize
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

# Compute Average Intra-Class Euclidean Distances
X = tr.X.tondarray()  # Convert CArray to NumPy array
y = tr.Y.tondarray()

unique_classes = np.unique(y)
intra_class_distances = {}

for cls in unique_classes:
    class_points = X[y == cls]  # Select only points from the given class
    if len(class_points) > 1:  # Avoid errors with single-point classes
        distances = pdist(class_points, metric='euclidean')
        avg_distance = np.mean(distances)
        intra_class_distances[cls] = avg_distance

# Print results
for cls, avg_dist in intra_class_distances.items():
    print(f"Class {cls}: Average intra-class Euclidean distance = {avg_dist:.4f}")
