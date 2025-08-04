import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from secml.array import CArray
from data.mnist_loader import load_mnist
from data.cifar_loader import load_data
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack

def cross_validate_knn_defense(X, y, n_splits=5, n_neighbors_ratio=0.1):
    """
    KNN defense için cross-validation
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Threshold arama aralığı
    thresholds = np.arange(0, 10, 0.1)
    cv_scores = {threshold: {'precision': [], 'recall': [], 'f1': [], 'accuracy': []} 
                 for threshold in thresholds}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # KNN hesaplama
        n_neighbors = int(len(X_train) * n_neighbors_ratio)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
        distances, indices = nbrs.kneighbors(X_val)
        
        # Ortalama uzaklık hesaplama
        average_distances = np.mean(distances, axis=1)
        
        # Her threshold için performans hesaplama
        for threshold in thresholds:
            # Poisoning tespiti
            poisoning_indices = np.where(average_distances < threshold)[0]
            
            # Predicted labels
            predicted_labels = np.zeros(len(X_val), dtype=int)
            predicted_labels[poisoning_indices] = 1
            
            # True labels (validation set'teki poisoning noktaları)
            true_labels = np.zeros(len(X_val), dtype=int)
            # Burada gerçek poisoning etiketlerini kullanmalısınız
            
            # Metrikler
            precision = precision_score(true_labels, predicted_labels, zero_division=0)
            recall = recall_score(true_labels, predicted_labels, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, zero_division=0)
            accuracy = accuracy_score(true_labels, predicted_labels)
            
            cv_scores[threshold]['precision'].append(precision)
            cv_scores[threshold]['recall'].append(recall)
            cv_scores[threshold]['f1'].append(f1)
            cv_scores[threshold]['accuracy'].append(accuracy)
    
    # En iyi threshold'u bulma (F1-score'a göre)
    mean_f1_scores = {threshold: np.mean(scores['f1']) for threshold, scores in cv_scores.items()}
    best_threshold = max(mean_f1_scores, key=mean_f1_scores.get)
    
    return best_threshold, cv_scores

def cross_validate_distance_defense(X, y, n_splits=5):
    """
    Distance-based defense için cross-validation
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    thresholds = np.arange(0.1, 12, 0.1)
    cv_scores = {threshold: {'precision': [], 'recall': [], 'f1': [], 'accuracy': []} 
                 for threshold in thresholds}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Sınıf 1'in ortalama noktasını hesaplama
        class_1_indices = np.where(y_train == 1)[0]
        class_1_points = X_train[class_1_indices]
        mean_point = np.mean(class_1_points, axis=0)
        
        # Sınıf 0 noktalarının uzaklıklarını hesaplama
        class_0_indices = np.where(y_val == 0)[0]
        class_0_points = X_val[class_0_indices]
        distances_to_mean = cdist(class_0_points, mean_point.reshape(1, -1), metric="euclidean").flatten()
        
        # Her threshold için performans hesaplama
        for threshold in thresholds:
            detected_indices = class_0_indices[distances_to_mean < threshold]
            
            predicted_labels = np.zeros(len(X_val), dtype=int)
            predicted_labels[detected_indices] = 1
            
            # True labels (validation set'teki poisoning noktaları)
            true_labels = np.zeros(len(X_val), dtype=int)
            # Burada gerçek poisoning etiketlerini kullanmalısınız
            
            # Metrikler
            precision = precision_score(true_labels, predicted_labels, zero_division=0)
            recall = recall_score(true_labels, predicted_labels, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, zero_division=0)
            accuracy = accuracy_score(true_labels, predicted_labels)
            
            cv_scores[threshold]['precision'].append(precision)
            cv_scores[threshold]['recall'].append(recall)
            cv_scores[threshold]['f1'].append(f1)
            cv_scores[threshold]['accuracy'].append(accuracy)
    
    # En iyi threshold'u bulma
    mean_f1_scores = {threshold: np.mean(scores['f1']) for threshold, scores in cv_scores.items()}
    best_threshold = max(mean_f1_scores, key=mean_f1_scores.get)
    
    return best_threshold, cv_scores

def cross_validate_kmeans_defense(X, y, n_splits=5, k_range=range(3, 10)):
    """
    K-Means defense için cross-validation
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = {k: {'precision': [], 'recall': [], 'f1': [], 'accuracy': []} for k in k_range}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Sınıf 1'in ortalama noktasını hesaplama
        class_1_indices = np.where(y_train == 1)[0]
        class_1_points = X_train[class_1_indices]
        mean_point = np.mean(class_1_points, axis=0)
        
        # Sınıf 0 noktalarının uzaklıklarını hesaplama
        class_0_indices = np.where(y_val == 0)[0]
        class_0_points = X_val[class_0_indices]
        distances_to_mean = cdist(class_0_points, mean_point.reshape(1, -1), metric='euclidean').flatten()
        
        sorted_indices = np.argsort(distances_to_mean)
        sorted_distances = distances_to_mean[sorted_indices].reshape(-1, 1)
        
        # Her k değeri için performans hesaplama
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(sorted_distances)
            kmeans_labels = kmeans.labels_
            
            # En küçük ortalama uzaklığa sahip cluster'ı seçme
            cluster_mean_distances = [np.mean(sorted_distances[kmeans_labels == i]) for i in range(k)]
            min_mean_cluster = np.argmin(cluster_mean_distances)
            detected_indices = class_0_indices[sorted_indices[kmeans_labels == min_mean_cluster]]
            
            predicted_labels = np.zeros(len(X_val), dtype=int)
            predicted_labels[detected_indices] = 1
            
            # True labels
            true_labels = np.zeros(len(X_val), dtype=int)
            # Burada gerçek poisoning etiketlerini kullanmalısınız
            
            # Metrikler
            precision = precision_score(true_labels, predicted_labels, zero_division=0)
            recall = recall_score(true_labels, predicted_labels, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, zero_division=0)
            accuracy = accuracy_score(true_labels, predicted_labels)
            
            cv_scores[k]['precision'].append(precision)
            cv_scores[k]['recall'].append(recall)
            cv_scores[k]['f1'].append(f1)
            cv_scores[k]['accuracy'].append(accuracy)
    
    # En iyi k değerini bulma
    mean_f1_scores = {k: np.mean(scores['f1']) for k, scores in cv_scores.items()}
    best_k = max(mean_f1_scores, key=mean_f1_scores.get)
    
    return best_k, cv_scores

def run_defense_with_cv(loader, dataset_name, digits_or_labels, defense_type='knn'):
    """
    Cross-validation ile defense çalıştırma
    """
    print(f"Running {defense_type} defense with cross-validation on {dataset_name}...")
    
    # Veri yükleme
    if dataset_name == "MNIST":
        tr, val, ts = loader(digits=digits_or_labels, n_tr=300, n_val=300, n_ts=300)
    elif dataset_name == "CIFAR":
        tr, val, ts = loader(labels=digits_or_labels, n_tr=300, n_val=300, n_ts=300)
    
    # Sınıflandırıcı eğitimi
    clf = LogisticClassifier()
    clf.init_fit(tr, {"C": 1})
    
    # Poisoning parametreleri
    params = {
        "n_proto": 30,
        "lb": 1,
        "y_target": np.array([1]),
        "y_poison": np.array([0]),
        "transform": lambda x: x,
    }
    poisoning_points, x_proto = run_attack(beta_poison, "beta_poison_attack", clf, tr, val, ts, params)
    
    # Veri hazırlama
    tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
    tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y
    
    all_x_poison = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] 
                                   for point in poisoning_points])
    all_y_poison = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] 
                                   for point in poisoning_points])
    
    sample_size = int(tr.X.shape[0] * 0.20)
    random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
    x_poison_all = all_x_poison[random_indices]
    y_poison_all = all_y_poison[random_indices]
    
    all_X = np.concatenate([tr_X, x_poison_all])
    all_Y = np.concatenate([tr_Y, y_poison_all])
    
    # Cross-validation ile optimal parametre bulma
    if defense_type == 'knn':
        best_param, cv_scores = cross_validate_knn_defense(all_X, all_Y)
        print(f"Best threshold: {best_param:.3f}")
    elif defense_type == 'distance':
        best_param, cv_scores = cross_validate_distance_defense(all_X, all_Y)
        print(f"Best threshold: {best_param:.3f}")
    elif defense_type == 'kmeans':
        best_param, cv_scores = cross_validate_kmeans_defense(all_X, all_Y)
        print(f"Best k: {best_param}")
    
    return best_param, cv_scores

# Kullanım örneği
if __name__ == "__main__":
    # MNIST için KNN defense
    best_threshold, scores = run_defense_with_cv(load_mnist, "MNIST", (4, 6), 'knn')
    
    # CIFAR için Distance defense
    best_threshold, scores = run_defense_with_cv(load_data, "CIFAR", (0, 8), 'distance')
    
    # K-Means defense
    best_k, scores = run_defense_with_cv(load_mnist, "MNIST", (4, 6), 'kmeans') 