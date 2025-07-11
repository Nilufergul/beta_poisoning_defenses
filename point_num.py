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


tr, val, ts = load_data(labels=(1, 9), n_tr=600, n_val=600, n_ts=600)

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

first_poisoning = poisoning_points[0]
x_poison, y_poison = first_poisoning
first_proto = x_proto[0]

tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
print(f"this is tr_X: {len(tr_X)}")
x_poison_all = x_poison
#x_poison_all = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])
print(f"this is x_poison_all: {len(x_poison_all)}")
tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y
print(f"this is tr_Y: {len(tr_Y)}")
y_poison_all = y_poison
#y_poison_all = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])
print(f"this is y_poison_all: {len(y_poison_all)}")

all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])
print(f"this is x_all: {len(all_X)}")
print(f"this is y_all: {len(all_Y)}")

class_label1 = 1
class_indices1 = np.where(all_Y == class_label1)[0]
class_points1 = all_X[class_indices1]
print(f"this is class_points1: {len(class_points1)}")

class_label2 = 0 
class_indices2 = np.where(all_Y == class_label2)[0]
class_points2 = all_X[class_indices2]
print(f"this is class_points2: {len(class_points2)}")