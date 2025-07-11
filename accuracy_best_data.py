import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from secml.array import CArray
from data.mnist_loader import load_mnist
from data.cifar_loader import load_data
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack

def run_distance_defense_with_threshold(loader, dataset_name, digits_or_labels, global_threshold=None):
    print(f"Running on {dataset_name}...")
    # Load data
    if dataset_name == "MNIST":
        tr, val, ts = loader(digits=digits_or_labels, n_tr=300, n_val=300, n_ts=300)
    elif dataset_name == "CIFAR":
        tr, val, ts = loader(labels=digits_or_labels, n_tr=300, n_val=300, n_ts=300)

    # Train classifier
    clf = LogisticClassifier()
    clf.init_fit(tr, {"C": 1})

    # Poisoning parameters
    params = {
        "n_proto": 30,
        "lb": 1,
        "y_target": np.array([1]),
        "y_poison": np.array([0]),
        "transform": lambda x: x,
    }
    poisoning_points, x_proto = run_attack(beta_poison, "beta_poison_attack", clf, tr, val, ts, params)

    # Prepare training data
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

    true_poisoning_indices = np.where(np.isin(all_X, x_poison_all).all(axis=1))[0]
    true_labels = np.zeros(len(all_X), dtype=int)
    true_labels[true_poisoning_indices] = 1

    # Calculate distances
    class_label1 = 1
    class_indices1 = np.where(all_Y == class_label1)[0]
    class_points1 = all_X[class_indices1]

    class_label2 = 0
    class_indices2 = np.where(all_Y == class_label2)[0]
    class_points2 = all_X[class_indices2]

    mean_point = np.mean(class_points1, axis=0)
    distances_to_mean = cdist(class_points2, mean_point.reshape(1, -1), metric="euclidean").flatten()

    # If a global threshold is provided, use it directly
    if global_threshold is not None:
        detected_poisoning_indices = class_indices2[distances_to_mean < global_threshold]
        predicted_labels = np.zeros(len(all_X), dtype=int)
        predicted_labels[detected_poisoning_indices] = 1

        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        accuracy = accuracy_score(true_labels, predicted_labels)

        return (precision, recall, f1, accuracy), global_threshold

    # Otherwise, find the best threshold
    thresholds = np.arange(0.1, 12, 0.1)
    best_f1, best_metrics, best_threshold = 0, None, None

    for threshold_distance in thresholds:
        detected_poisoning_indices = class_indices2[distances_to_mean < threshold_distance]
        predicted_labels = np.zeros(len(all_X), dtype=int)
        predicted_labels[detected_poisoning_indices] = 1

        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        accuracy = accuracy_score(true_labels, predicted_labels)

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = (precision, recall, f1, accuracy)
            best_threshold = threshold_distance

    return best_metrics, best_threshold

# Dataset loaders
dataset_loaders = {
    "MNIST": load_mnist,
    "CIFAR": load_data,
}

# Digits or labels for datasets
datasets = {
    "MNIST": {"loader": dataset_loaders["MNIST"], "digits_or_labels": (4, 6)},
    "CIFAR": {"loader": dataset_loaders["CIFAR"], "digits_or_labels": (0, 8)},
}

# Determine the best global threshold
all_thresholds = []
for dataset_name, params in datasets.items():
    _, threshold = run_distance_defense_with_threshold(params["loader"], dataset_name, params["digits_or_labels"])
    all_thresholds.append(threshold)

global_threshold = max(all_thresholds)  # Use the maximum threshold for both datasets

# Run the defense using the global threshold
data = []
for dataset_name, params in datasets.items():
    best_metrics, _ = run_distance_defense_with_threshold(params["loader"], dataset_name, params["digits_or_labels"], global_threshold=global_threshold)
    data.extend(best_metrics)  # Append metrics to the data array

# Print results
print("Final Data Array (Precision(MNIST), Precision(CIFAR), Recall(MNIST), Recall(CIFAR), ...):")
print(data)
print(f"Global Threshold Used: {global_threshold}")
