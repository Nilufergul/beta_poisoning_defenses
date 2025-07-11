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
x_poison_all = x_poison

tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y
y_poison_all = y_poison

# Tüm verileri birleştir
all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])

# Class 1 ve Class 9 olarak verileri ayır
class_label1 = 0  
class_indices1 = np.where(all_Y == class_label1)[0]
class_points1 = all_X[class_indices1]

class_label2 = 1 
class_indices2 = np.where(all_Y == class_label2)[0]
class_points2 = all_X[class_indices2]

# Her iki sınıf için ortalama noktayı hesapla
mean_point_class1 = np.mean(class_points1, axis=0)
mean_point_class2 = np.mean(class_points2, axis=0)

# 1. Koşul: Class 1'de olup Class 2'nin ortalamasına uzak olan noktalar
distances_to_class2_mean = cdist(class_points1, mean_point_class2.reshape(1, -1), metric='euclidean').flatten()
sorted_distances_class2_indices = np.argsort(distances_to_class2_mean)
sorted_distances_class2 = distances_to_class2_mean[sorted_distances_class2_indices]

# Class 2'ye uzak olan outlier'ları tespit et
Q1_class2 = np.percentile(sorted_distances_class2, 25)
Q3_class2 = np.percentile(sorted_distances_class2, 75)
IQR_class2 = Q3_class2 - Q1_class2
threshold_class2 = Q1_class2 - 1.5 * IQR_class2

outliers_class2 = class_indices1[distances_to_class2_mean < threshold_class2]

# 2. Koşul: Class 1'in kendi ortalamasına uzak olan noktalar
distances_to_class1_mean = cdist(class_points1, mean_point_class1.reshape(1, -1), metric='euclidean').flatten()
sorted_distances_class1_indices = np.argsort(distances_to_class1_mean)
sorted_distances_class1 = distances_to_class1_mean[sorted_distances_class1_indices]

print(sorted_distances_class1)
print(sorted_distances_class2)

# Class 1'den kendi ortalamasına uzak olan outlier'ları tespit et
Q1_class1 = np.percentile(sorted_distances_class1, 25)
Q3_class1 = np.percentile(sorted_distances_class1, 75)
IQR_class1 = Q3_class1 - Q1_class1
threshold_class1 = Q3_class1 + 1.5 * IQR_class1

outliers_class1 = class_indices1[distances_to_class1_mean > threshold_class1]

# İki koşulu sağlayan noktaları bulun
detected_poisoning_indices = np.intersect1d(outliers_class2, outliers_class1)

# True poisoning point'leri belirle
true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

poisoning_points_distances_to_mean = cdist(all_X[true_poisoning_indices], mean_point_class1.reshape(1, -1), metric='euclidean').flatten()

# En küçükten büyüğe sıralama
sorted_poisoning_distances = np.sort(poisoning_points_distances_to_mean)
print("\nSıralanmış Poisoning Noktalarının Class 1'in(kendisinin) Ortalamasına Olan Uzaklıkları:")
print(sorted_poisoning_distances)

poisoning_points_distances_to_mean = cdist(all_X[true_poisoning_indices], mean_point_class2.reshape(1, -1), metric='euclidean').flatten()

# En küçükten büyüğe sıralama
sorted_poisoning_distances = np.sort(poisoning_points_distances_to_mean)
print("\nSıralanmış Poisoning Noktalarının Class 2'in Ortalamasına Olan Uzaklıkları:")
print(sorted_poisoning_distances)

# İki sınıfın ortalama noktaları arasındaki Öklidyen mesafeyi hesapla
mean_distance = np.linalg.norm(mean_point_class1 - mean_point_class2)
print(f"Class 1'in ortalama noktası ile Class 2'nin ortalama noktası arasındaki mesafe: {mean_distance:.4f}")


# Binary true labels (1 for real poisoning points, 0 for others)
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

# Binary predicted labels (1 for detected poisoning points, 0 for others)
predicted_labels = np.zeros(len(all_X), dtype=int)
predicted_labels[detected_poisoning_indices] = 1

# Performans metriklerini hesapla
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(all_X)

# Görselleştirme
plt.figure(figsize=(12, 10))

# Üstteki scatter plot
plt.subplot(2, 1, 1)

# Class 1 ve Class 9 verilerini çiz
for i, label in enumerate(np.unique(all_Y)):
    indices = np.where(all_Y == label)
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], marker='o', label=f'Class {label}', alpha=0.5, s=50)

# Poisoning point olarak işaretlenen noktaları çiz
if len(detected_poisoning_indices) > 0:
    plt.scatter(X_reduced[detected_poisoning_indices, 0], X_reduced[detected_poisoning_indices, 1], 
                c='red', marker='o', label='Detected Poisoned Points', s=50)

# Mean noktaları ekleyelim
mean_point_class1_reduced = pca.transform(mean_point_class1.reshape(1, -1))
mean_point_class2_reduced = pca.transform(mean_point_class2.reshape(1, -1))
plt.scatter(mean_point_class1_reduced[:, 0], mean_point_class1_reduced[:, 1], c='cyan', marker='X', label='Mean Point Class 1', s=50)
plt.scatter(mean_point_class2_reduced[:, 0], mean_point_class2_reduced[:, 1], c='magenta', marker='X', label='Mean Point Class 2', s=50)

plt.legend()
plt.title('Detected Poisoning Points and Mean Points')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# Alt kısımda performans metrikleri
plt.subplot(2, 1, 2)
plt.text(0.5, 0.5, f'Threshold Class 1: {threshold_class1}\n'
                   f'Threshold Class 2: {threshold_class2}\n'
                   f'Precision: {precision}\n'
                   f'Recall: {recall}\n'
                   f'F1-Score: {f1}\n'
                   f'Accuracy: {accuracy}', 
         horizontalalignment='center', verticalalignment='center', fontsize=15)
plt.axis('off')

plt.tight_layout()
plt.savefig('z_get_threshold/two_condition/cifar_threshold_1-9.png')
plt.show()