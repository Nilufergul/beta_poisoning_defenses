import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from secml.array import CArray
from data.mnist_loader import *
from data.cifar_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack


tr, val, ts = load_data(labels=(1, 9), n_tr=300, n_val=300, n_ts=300)

clf = LogisticClassifier()
clf.init_fit(tr, {"C": 1})


params = {
    "n_proto": 15,
    "lb": 1,
    "y_target": np.array([1]),
    "y_poison": np.array([0]),
    "transform": lambda x: x,
}


poisoning_points, x_proto = run_attack(beta_poison, "beta_poison_attack", clf, tr, val, ts, params)

print(poisoning_points)
num_poisoning_points = len(poisoning_points)

print(f"Zehirlenmiş veri noktalarının sayısı: {num_poisoning_points}")
first_poisoning = poisoning_points[0]

print(first_poisoning)
first_poisoning_points = len(first_poisoning)

print(f"Zehirlenmiş veri noktalarının sayısı: {first_poisoning_points}")
x_poison, y_poison = first_poisoning
first_proto = x_proto[0]

tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
x_poison_all = x_poison
#x_poison_all = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])

tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y
y_poison_all = y_poison
#y_poison_all = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])