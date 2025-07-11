import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from secml.array import CArray

from data.mnist_loader import load_mnist
from data.cifar_loader import load_data
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack

# MNIST verisini yükle
tr, val, ts = load_mnist(digits=(4, 6), n_tr=300, n_val=300, n_ts=300)

# Sınıflandırıcıyı başlat ve eğit
clf = LogisticClassifier()
clf.init_fit(tr, {"C": 1})

# Saldırı parametrelerini ayarla
params = {
    "n_proto": 30,
    "lb": 1,
    "y_target": np.array([1]),
    "y_poison": np.array([0]),
    "transform": lambda x: x,  
}

# Poisoning saldırısını çalıştır
poisoning_points, x_proto = run_attack(beta_poison, "beta_poison_attack", clf, tr, val, ts, params)

# Eğitim verilerini numpy dizisine dönüştür
tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y

# Poisoning noktalarını birleştir
all_x_poison = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])
all_y_poison = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])

# Poisoning noktalarından rastgele bir alt küme seç
sample_size = int(tr_X.shape[0] * 0.20)  # Eğitim setinin %20'si
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


# Poisoning ve prototip noktalarını PCA ile görselleştir
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(tr_X)
poisoned_X_reduced = pca.transform(x_poison_all)
mean_point_reduced = pca.transform(mean_point.reshape(1, -1))



if __name__ == "__main__":
    plt.rcParams['legend.fontsize'] = 12
    plt.figure(figsize=(6, 4))
    colors = ['magenta', 'blue'] 
    
    # Sınıfları görselleştir
    for i, label in enumerate(np.unique(tr.Y.tondarray())):
        indices = np.where(tr.Y.tondarray() == label)
        plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], marker='o', label=f'Class {label}', c=colors[i], alpha=0.7, s=75)
    
    # Poisoning noktaları ve prototipleri ekle
    plt.scatter(poisoned_X_reduced[:, 0], poisoned_X_reduced[:, 1], c='red', marker='x', label='Poisoning Points', s=150)
    plt.scatter(mean_point_reduced[:, 0], mean_point_reduced[:, 1], color='cyan', marker='X', label='Mean Point of Class 1', s=250)

    plt.legend()
    plt.title('MNIST Data Set with Poisoning Points', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)

    plt.tight_layout()
    plt.savefig('/Users/nilufergulciftci/desktop/poisoning/mnist_poisoned_points_first.pdf')
    plt.show()
