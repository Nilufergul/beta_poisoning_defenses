import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from data.mnist_loader import *
from data.cifar_loader import *
from secml.array import CArray
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack

# Veri yükleme ve sınıflandırıcı eğitimi
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
thresholds = np.arange(0.1, 12, 0.2) # 0.1'den 10'a kadar 0.1 artışlarla eşik değerleri
roc_point = []

for threshold_distance in thresholds:
    # Eşik mesafeye göre zehirlenmiş noktaların tespiti
    detected_poisoning_indices = class_indices2[distances_to_mean < threshold_distance]

    # Binary predicted labels (1 for detected poisoning points, 0 for others)
    predicted_labels = np.zeros(len(all_X), dtype=int)
    predicted_labels[detected_poisoning_indices] = 1

    tp = np.sum((predicted_labels == 1) & (true_labels == 1))
    tn = np.sum((predicted_labels == 0) & (true_labels == 0))
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))

    # TPR ve FPR hesapla
    tpr = tp / (tp + fn)  # True Positive Rate
    fpr = tp / (tp + fp)  # False Positive Rate

    roc_point.append([tpr, fpr])

pivot = pd.DataFrame(roc_point, columns = ["x", "y"])
plt.scatter(pivot.y, pivot.x)
plt.plot([0,1])
plt.savefig('/Users/nilufergulciftci/desktop/graphs/roc_curve.png')
plt.show()

#import matplotlib.pyplot as plt
#import seaborn as sns

# Pozitif (zehirlenmiş) ve negatif (temiz) mesafeleri ayırma
#poisoned_distances = distances_to_mean[true_labels[class_indices2] == 1]  # Zehirlenmiş örnekler
#clean_distances = distances_to_mean[true_labels[class_indices2] == 0]     # Temiz örnekler

# Histogram veya KDE çizimi ile mesafe dağılımlarını görselleştirme
#plt.figure(figsize=(10, 6))
#sns.kdeplot(poisoned_distances, label="Poisoned Distances", shade=True)
#sns.kdeplot(clean_distances, label="Clean Distances", shade=True)
#plt.xlabel("Distance to Mean")
#plt.ylabel("Density")
#plt.title("Distance Distributions for Poisoned and Clean Examples")
#plt.legend()
#plt.show()