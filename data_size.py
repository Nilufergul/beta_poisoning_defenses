import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from secml.array import CArray
from data.mnist_loader import *
from data.cifar_loader import *
from data.cifar100_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

f1_results = {"KPB": [], "NCC": [], "CBD": [], "MDT": []}
precision = {"KPB": [], "NCC": [], "CBD": [], "MDT": []}
recall = {"KPB": [], "NCC": [], "CBD": [], "MDT": []}
accuracy = {"KPB": [], "NCC": [], "CBD": [], "MDT": []}

tr, val, ts = load_data100(labels=(7, 15), n_tr=300, n_val=300, n_ts=300)
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

# Zaman içinde oluşturulan poisoning noktalarını elde ediyoruz
all_x_poison = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])
all_y_poison = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])

sample_size = int(tr.X.shape[0] * 0.20)
random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
x_poison_all = all_x_poison[random_indices]
y_poison_all = all_y_poison[random_indices]

# Tüm training verisi (temiz ve zehirli) birleşimi
all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])
# Orijinal veriyi korumak için kopyalayalım
orig_X = all_X.copy()
orig_Y = all_Y.copy()

### KPB Savunması (kNN Proximity-Based Defense)
X_kpb = orig_X.copy()
Y_kpb = orig_Y.copy()

n_neighbors_kpb = int(len(X_kpb) * 0.1)
nbrs_kpb = NearestNeighbors(n_neighbors=n_neighbors_kpb).fit(X_kpb)
distances_kpb, _ = nbrs_kpb.kneighbors(X_kpb)
average_distances_kpb = np.mean(distances_kpb, axis=1)
poisoning_indices_kpb = np.where(average_distances_kpb < 4)[0]

true_poisoning_indices_kpb = np.where(np.isin(X_kpb, x_poison_all).all(axis=1))[0]
true_labels_kpb = np.zeros(len(X_kpb), dtype=int)
true_labels_kpb[true_poisoning_indices_kpb] = 1

predicted_labels_kpb = np.zeros(len(X_kpb), dtype=int)
predicted_labels_kpb[poisoning_indices_kpb] = 1

tp_kpb = np.sum((true_labels_kpb == 1) & (predicted_labels_kpb == 1))
fp_kpb = np.sum((true_labels_kpb == 0) & (predicted_labels_kpb == 1))
fn_kpb = np.sum((true_labels_kpb == 1) & (predicted_labels_kpb == 0))
tn_kpb = np.sum((true_labels_kpb == 0) & (predicted_labels_kpb == 0))
accuracy_kpb = (tp_kpb + tn_kpb) / len(true_labels_kpb)
precision_kpb = tp_kpb / (tp_kpb + fp_kpb) if (tp_kpb + fp_kpb) > 0 else 0
recall_kpb = tp_kpb / (tp_kpb + fn_kpb) if (tp_kpb + fn_kpb) > 0 else 0
f1_kpb = (2 * precision_kpb * recall_kpb) / (precision_kpb + recall_kpb) if (precision_kpb + recall_kpb) > 0 else 0
f1_results["KPB"].append(f1_kpb)
precision["KPB"].append(precision_kpb)
recall["KPB"].append(recall_kpb)
accuracy["KPB"].append(accuracy_kpb)

### NCC Savunması (Neighborhood Class Comparison)
X_ncc = orig_X.copy()
Y_ncc = orig_Y.copy()

knn_ncc = NearestNeighbors(n_neighbors=(len(y_poison_all) * 2))
knn_ncc.fit(X_ncc)
distances_ncc, indices_ncc = knn_ncc.kneighbors(X_ncc)

detected_poisoning_indices_ncc = []
poisoning_count_ncc = len(y_poison_all)
inner_neighbors_ncc = poisoning_count_ncc
outer_neighbors_ncc = poisoning_count_ncc + int(poisoning_count_ncc * 0.2)
for i, neighbors in enumerate(indices_ncc):
    inner = Y_ncc[neighbors[:inner_neighbors_ncc]]
    outer = Y_ncc[neighbors[inner_neighbors_ncc:outer_neighbors_ncc]]
    most_common_inner_class_ncc = np.bincount(inner).argmax()
    most_common_outer_class_ncc = np.bincount(outer).argmax()
    if most_common_inner_class_ncc != most_common_outer_class_ncc:
        detected_poisoning_indices_ncc.append(i)

true_poisoning_indices_ncc = np.where(np.isin(X_ncc, x_poison_all).all(axis=1))[0]
true_labels_ncc = np.zeros(len(X_ncc), dtype=int)
true_labels_ncc[true_poisoning_indices_ncc] = 1
predicted_labels_ncc = np.zeros(len(X_ncc), dtype=int)
predicted_labels_ncc[detected_poisoning_indices_ncc] = 1

tp_ncc = np.sum((true_labels_ncc == 1) & (predicted_labels_ncc == 1))
fp_ncc = np.sum((true_labels_ncc == 0) & (predicted_labels_ncc == 1))
fn_ncc = np.sum((true_labels_ncc == 1) & (predicted_labels_ncc == 0))
tn_ncc = np.sum((true_labels_ncc == 0) & (predicted_labels_ncc == 0))
accuracy_ncc = (tp_ncc + tn_ncc) / len(true_labels_ncc)
precision_ncc = tp_ncc / (tp_ncc + fp_ncc) if (tp_ncc + fp_ncc) > 0 else 0
recall_ncc = tp_ncc / (tp_ncc + fn_ncc) if (tp_ncc + fn_ncc) > 0 else 0
f1_ncc = (2 * precision_ncc * recall_ncc) / (precision_ncc + recall_ncc) if (precision_ncc + recall_ncc) > 0 else 0
f1_results["NCC"].append(f1_ncc)
precision["NCC"].append(precision_ncc)
recall["NCC"].append(recall_ncc)
accuracy["NCC"].append(accuracy_ncc)

### CBD Savunması (Clustering-Based Defense)
X_cbd = orig_X.copy()
Y_cbd = orig_Y.copy()

class_indices1_cbd = np.where(Y_cbd == 1)[0]
class_points1_cbd = X_cbd[class_indices1_cbd]
class_indices2_cbd = np.where(Y_cbd == 0)[0]
class_points2_cbd = X_cbd[class_indices2_cbd]

mean_point_cbd = np.mean(class_points1_cbd, axis=0)
distances_to_mean_cbd = cdist(class_points2_cbd, mean_point_cbd.reshape(1, -1), metric='euclidean').flatten()
sorted_indices_cbd = np.argsort(distances_to_mean_cbd)
sorted_distances_cbd = distances_to_mean_cbd[sorted_indices_cbd].reshape(-1, 1)

k_values_cbd = range(3, 10)
sse_cbd = []
random_state_cbd = 0
np.random.seed(random_state_cbd)

for k in k_values_cbd:
    kmeans_cbd = KMeans(n_clusters=k, random_state=random_state_cbd)
    kmeans_cbd.fit(sorted_distances_cbd)
    sse_cbd.append(kmeans_cbd.inertia_)

optimal_k_cbd = k_values_cbd[np.argmin(np.gradient(sse_cbd))]
kmeans_cbd = KMeans(n_clusters=optimal_k_cbd, random_state=random_state_cbd)
kmeans_cbd.fit(sorted_distances_cbd)
kmeans_labels_cbd = kmeans_cbd.labels_
cluster_mean_distances_cbd = [np.mean(sorted_distances_cbd[kmeans_labels_cbd == i]) for i in range(optimal_k_cbd)]
min_mean_cluster_cbd = np.argmin(cluster_mean_distances_cbd)
detected_poisoning_indices_cbd = np.where(Y_cbd == 0)[0][sorted_indices_cbd[kmeans_labels_cbd == min_mean_cluster_cbd]]

true_poisoning_indices_cbd = np.where(np.isin(X_cbd, x_poison_all).all(axis=1))[0]
true_labels_cbd = np.zeros(len(X_cbd), dtype=int)
true_labels_cbd[true_poisoning_indices_cbd] = 1
predicted_labels_cbd = np.zeros(len(X_cbd), dtype=int)
predicted_labels_cbd[detected_poisoning_indices_cbd] = 1

tp_cbd = np.sum((true_labels_cbd == 1) & (predicted_labels_cbd == 1))
fp_cbd = np.sum((true_labels_cbd == 0) & (predicted_labels_cbd == 1))
fn_cbd = np.sum((true_labels_cbd == 1) & (predicted_labels_cbd == 0))
tn_cbd = np.sum((true_labels_cbd == 0) & (predicted_labels_cbd == 0))
accuracy_cbd = (tp_cbd + tn_cbd) / len(true_labels_cbd)
precision_cbd = tp_cbd / (tp_cbd + fp_cbd) if (tp_cbd + fp_cbd) > 0 else 0
recall_cbd = tp_cbd / (tp_cbd + fn_cbd) if (tp_cbd + fn_cbd) > 0 else 0
f1_cbd = (2 * precision_cbd * recall_cbd) / (precision_cbd + recall_cbd) if (precision_cbd + recall_cbd) > 0 else 0
f1_results["CBD"].append(f1_cbd)
precision["CBD"].append(precision_cbd)
recall["CBD"].append(recall_cbd)
accuracy["CBD"].append(accuracy_cbd)

### MDT Savunması (Mean Distance Threshold Defense)
X_mdt = orig_X.copy()
Y_mdt = orig_Y.copy()

class_indices1_mdt = np.where(Y_mdt == 1)[0]
class_points1_mdt = X_mdt[class_indices1_mdt]
class_indices2_mdt = np.where(Y_mdt == 0)[0]
class_points2_mdt = X_mdt[class_indices2_mdt]

mean_point_mdt = np.mean(class_points1_mdt, axis=0)
distances_to_mean_mdt = cdist(class_points2_mdt, mean_point_mdt.reshape(1, -1), metric='euclidean').flatten()

true_poisoning_indices_mdt = np.where(np.isin(X_mdt, x_poison_all).all(axis=1))[0]
true_labels_mdt = np.zeros(len(X_mdt), dtype=int)
true_labels_mdt[true_poisoning_indices_mdt] = 1

detected_poisoning_indices_mdt = np.where(distances_to_mean_mdt < 6)[0]
# Map back to original indices for class_points2_mdt
detected_poisoning_indices_mdt = class_indices2_mdt[detected_poisoning_indices_mdt]

predicted_labels_mdt = np.zeros(len(X_mdt), dtype=int)
predicted_labels_mdt[detected_poisoning_indices_mdt] = 1

tp_mdt = np.sum((true_labels_mdt == 1) & (predicted_labels_mdt == 1))
fp_mdt = np.sum((true_labels_mdt == 0) & (predicted_labels_mdt == 1))
fn_mdt = np.sum((true_labels_mdt == 1) & (predicted_labels_mdt == 0))
tn_mdt = np.sum((true_labels_mdt == 0) & (predicted_labels_mdt == 0))
accuracy_mdt = (tp_mdt + tn_mdt) / len(true_labels_mdt)
precision_mdt = tp_mdt / (tp_mdt + fp_mdt) if (tp_mdt + fp_mdt) > 0 else 0
recall_mdt = tp_mdt / (tp_mdt + fn_mdt) if (tp_mdt + fn_mdt) > 0 else 0
f1_mdt = (2 * precision_mdt * recall_mdt) / (precision_mdt + recall_mdt) if (precision_mdt + recall_mdt) > 0 else 0
f1_results["MDT"].append(f1_mdt)
precision["MDT"].append(precision_mdt)
recall["MDT"].append(recall_mdt)
accuracy["MDT"].append(accuracy_mdt)

print(f1_results["KPB"])
print(f1_results["NCC"])
print(f1_results["CBD"])
print(f1_results["MDT"])
print("\n")

print(precision["KPB"])
print(precision["NCC"])
print(precision["CBD"])
print(precision["MDT"])
print("\n")

print(recall["KPB"])
print(recall["NCC"])
print(recall["CBD"])
print(recall["MDT"])
print("\n")


print(accuracy["KPB"])
print(accuracy["NCC"])
print(accuracy["CBD"])
print(accuracy["MDT"])
print("\n")

