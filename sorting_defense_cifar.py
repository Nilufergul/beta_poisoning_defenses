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

# Veriyi yükleyelim
tr, val, ts = load_data(labels=(0, 1), n_tr=100, n_val=400, n_ts=600)

# Modeli eğitmek için LogisticClassifier kullanılıyor
clf = LogisticClassifier()
clf.init_fit(tr, {"C": 1})

# Parametreleri tanımla
params = {
    "n_proto": 30,
    "lb": 1,
    "y_target": np.array([1]),
    "y_poison": np.array([0]),
    "transform": lambda x: x,
}

# Poisoning attack çalıştır
poisoning_points, x_proto = run_attack(beta_poison, "beta_poison_attack", clf, tr, val, ts, params)

# İlk poisoning ve proto noktalarını al
first_poisoning = poisoning_points[0]
x_poison, y_poison = first_poisoning
first_proto = x_proto[0]

# Veriyi ve etiketleri numpy dizilerine dönüştür
tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
ts_X = ts.X.tondarray() if isinstance(ts.X, CArray) else ts.X
x_poison_all = x_poison

tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y
ts_Y = ts.Y.tondarray() if isinstance(ts.Y, CArray) else ts.Y
y_poison_all = y_poison

# Tüm verileri birleştir
all_X = np.concatenate([tr_X, ts_X, x_poison_all])
all_Y = np.concatenate([tr_Y, ts_Y, y_poison_all])

# Class 1 ve Class 0 verilerini ayır
class_label1 = 1  
class_indices1 = np.where(all_Y == class_label1)[0]
class_points1 = all_X[class_indices1]

class_label2 = 0  
class_indices2 = np.where(all_Y == class_label2)[0]
class_points2 = all_X[class_indices2]

# Ortalama noktayı hesapla (class 1 için)
mean_point = np.mean(class_points1, axis=0)

# Class 0 (poisoned olabilecek) verilerin ortalama noktaya olan mesafelerini hesapla
distances_to_mean = cdist(class_points2, mean_point.reshape(1, -1), metric='euclidean').flatten()

# Mesafeleri küçükten büyüğe sırala
sorted_indices = np.argsort(distances_to_mean)
sorted_distances = distances_to_mean[sorted_indices]

# Mesafeler arasındaki farkları hesapla
distance_diffs = np.diff(sorted_distances)

# En büyük 3 farkın indekslerini bul
top_3_diff_indices = np.argsort(distance_diffs)[-3:]

# Farkların küçük olanlarına en yakın farkı seç
chosen_diff_index = top_3_diff_indices[np.argmin(sorted_distances[top_3_diff_indices])]

# Seçilen farkın olduğu mesafeye kadar olan veriler bir küme, kalan veriler diğer küme olacak
detected_poisoning_indices_sorted = class_indices2[sorted_indices[:chosen_diff_index + 1]]
remaining_indices_sorted = class_indices2[sorted_indices[chosen_diff_index + 1:]]

# Gerçek poisoning noktalarını belirle
true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]

# Binary true labels (1 for real poisoning points, 0 for others)
true_labels = np.zeros(len(all_X), dtype=int)
true_labels[true_poisoning_indices] = 1

# Binary predicted labels (1 for detected poisoning points, 0 for others)
predicted_labels_sorted = np.zeros(len(all_X), dtype=int)
predicted_labels_sorted[detected_poisoning_indices_sorted] = 1

# Sonuçları karşılaştır
precision_sorted = precision_score(true_labels, predicted_labels_sorted)
recall_sorted = recall_score(true_labels, predicted_labels_sorted)
f1_sorted = f1_score(true_labels, predicted_labels_sorted)
accuracy_sorted = accuracy_score(true_labels, predicted_labels_sorted)

print(f"Sorted Precision: {precision_sorted:.2f}")
print(f"Sorted Recall: {recall_sorted:.2f}")
print(f"Sorted F1-Score: {f1_sorted:.2f}")
print(f"Sorted Accuracy: {accuracy_sorted:.2f}")

# PCA ile görselleştirme
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(all_X)
mean_point_reduced = pca.transform(mean_point.reshape(1, -1))

plt.figure(figsize=(12, 8))

# Tüm sınıfları çiz
for i, label in enumerate(np.unique(all_Y)):
    indices = np.where(all_Y == label)
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], marker='o', label=f'Class {label}', alpha=0.5, s=50)

# Sıralama yöntemiyle tespit edilen zehirlenmiş noktaları çiz
plt.scatter(X_reduced[detected_poisoning_indices_sorted, 0], X_reduced[detected_poisoning_indices_sorted, 1], 
            c='red', marker='X', label='Detected Poisoned Points (Cluster 1)', s=20)

# Ortalama noktayı çiz
plt.scatter(mean_point_reduced[:, 0], mean_point_reduced[:, 1], c='magenta', marker='X', label='Mean Point', s=100)

plt.legend()
plt.title('Mesafeye Göre Sıralama ile Tespit Edilen Zehirlenmiş Noktalar ve Kümeleme')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
#plt.savefig('sorted_accuracy_cifar1-2.png')
plt.show()