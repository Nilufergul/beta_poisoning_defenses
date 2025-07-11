# Bu dosya karşı sınıfın mean'ine olan yakınlığına göre hesaplanan uzaklıkları, bir threshold değerine göre karşılaştırır.
# Bu karşılaştırmalara göre artan threshold değerleriyle accurcacy vs grafikleri çizer.


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

#tr, val, ts = load_mnist(digits=(4, 6), n_tr=300, n_val=300, n_ts=300)
tr, val, ts = load_data(labels=(0, 8), n_tr=300, n_val=300, n_ts=300)
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

sample_size = int(tr.X.shape[0] * 0.20)
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

true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

thresholds = np.arange(0.1, 12, 0.1)
precisions = []
recalls = []
f1_scores = []
accuracies = []

for threshold_distance in thresholds:
    detected_poisoning_indices = class_indices2[distances_to_mean < threshold_distance]

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

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

import matplotlib.ticker as mticker

figure, axis = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

def format_ticks(value, _):
    return f"{value:.3f}"

#plt.rcParams.update({'font.size': 16})
axis[0,0].plot(thresholds, precisions, label='CIFAR-10', color='blue')
axis[0,0].set_title('Precision',fontsize=22)
axis[0,0].set_xlabel('tau', fontsize=18)
axis[0,0].set_ylabel('Precision', fontsize=18)
axis[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axis[0,0].legend()

axis[0,1].plot(thresholds, recalls, label='CIFAR-10', color='blue')
axis[0,1].set_title('Recall',fontsize=22)
axis[0,1].set_xlabel('tau', fontsize=18)
axis[0,1].set_ylabel('Recall', fontsize=18)
axis[0,1].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axis[0,1].legend()

axis[1,0].plot(thresholds, f1_scores, label='CIFAR-10', color='blue')
axis[1,0].set_title('F1-Score',fontsize=22)
axis[1,0].set_xlabel('tau', fontsize=18)
axis[1,0].set_ylabel('F1-Score', fontsize=18)
axis[1,0].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axis[1,0].legend()

axis[1,1].plot(thresholds, accuracies, label='CIFAR-10', color='blue')
axis[1,1].set_title('Accuracy',fontsize=22)
axis[1,1].set_xlabel('tau', fontsize=18)
axis[1,1].set_ylabel('Accuracy', fontsize=18)
axis[1,1].yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
axis[1,1].legend()

plt.tight_layout()
plt.savefig('/Users/nilufergulciftci/desktop/poisoning/Defense4/mean_distance_graph_cifar_all.pdf')
plt.close()