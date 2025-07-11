
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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
# Assuming `run_attack` and other variables are already defined
poisoning_points, x_proto = run_attack(beta_poison, "beta_poison_attack", clf, tr, val, ts, params)

# Convert tr.X and tr.Y to numpy arrays if they are of type CArray
tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X
tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y

# Concatenate all poisoning points from the attack output
all_x_poison = np.concatenate([
    point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points
])
all_y_poison = np.concatenate([
    point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points
])

# Define the sample size as 20% of the training set
sample_size = int(tr_X.shape[0] * 0.20)

# Select a random sample of poisoned points
random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
x_poison_all = all_x_poison[random_indices]
y_poison_all = all_y_poison[random_indices]

# Calculate the mean of the non-poisoned class in the training dataset
non_poisoned_mean = np.mean(tr_X[tr_Y == params["y_target"]], axis=0)

# Concatenate all points (training and sampled poisoned points)
all_points = np.concatenate([tr_X[tr_Y != params["y_target"]], x_poison_all], axis=0)
all_labels = np.concatenate([np.zeros(len(tr_X[tr_Y != params["y_target"]])), np.ones(len(x_poison_all))])  # 0 for non-poisoned, 1 for poisoned

# Calculate distances from the non-poisoned mean for each point
distances = np.linalg.norm(all_points - non_poisoned_mean, axis=1)
distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

# Calculate ROC curve using distances as `y_score`
fpr, tpr, thresholds = roc_curve(all_labels, distances)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess Line")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Using Distances from Mean")
plt.legend()
plt.show()

# Optionally, you can find the optimal threshold (where TPR - FPR is maximized)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold for Distance:", optimal_threshold)

