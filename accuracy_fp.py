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

# Veri yükleme ve sınıflandırıcı eğitimi
#tr, val, ts = load_mnist(digits=(3, 8), n_tr=100, n_val=400, n_ts=600)
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

# Binary true labels (1 for real poisoning points, 0 for others)
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

# Threshold aralığı ve metrikler için boş listeler
thresholds = np.arange(0.1, 12, 0.1)  # 0.1'den 10'a kadar 0.1 artışlarla eşik değerleri
# Additional lists to store TP, TN, FP, FN values for each threshold
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []

for threshold_distance in thresholds:
    # Detect poisoning points based on the current threshold
    detected_poisoning_indices = class_indices2[distances_to_mean < threshold_distance]

    # Binary predicted labels (1 for detected poisoning points, 0 for others)
    predicted_labels = np.zeros(len(all_X), dtype=int)
    predicted_labels[detected_poisoning_indices] = 1

    # Calculate TP, TN, FP, FN
    tp = np.sum((predicted_labels == 1) & (true_labels == 1))
    tn = np.sum((predicted_labels == 0) & (true_labels == 0))
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))

    # Append values to respective lists
    true_positives.append(tp)
    true_negatives.append(tn)
    false_positives.append(fp)
    false_negatives.append(fn)

# Plotting TP, TN, FP, FN
plt.figure(figsize=(10, 6))
plt.plot(thresholds, true_positives, label='True Positives (TP)', linestyle='-')
plt.plot(thresholds, true_negatives, label='True Negatives (TN)', linestyle='--')
plt.plot(thresholds, false_positives, label='False Positives (FP)', linestyle='-.')
plt.plot(thresholds, false_negatives, label='False Negatives (FN)', linestyle=':')

plt.xlabel('Threshold (τ)')
plt.ylabel('Count')
plt.title('TP, TN, FP, FN Changes Across Threshold Values')
plt.legend()
plt.grid(True)
plt.savefig('/Users/nilufergulciftci/desktop/graphs/fp_fn_values.png')
plt.show()