import numpy as np

from secml.array import CArray
from data.mnist_loader import *
from src.classifier.secml_classifier import LogisticClassifier
from src.optimizer.beta_optimizer import beta_poison
from src.experiments.run_attack import run_attack


tr, val, ts = load_mnist(digits=(4,6), n_tr=300, n_val=300, n_ts=300)


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

#first_poisoning = poisoning_points[0]
#x_poison, y_poison = first_poisoning
#first_proto = x_proto[0]


tr_X = tr.X.tondarray() if isinstance(tr.X, CArray) else tr.X

#x_poison_all = x_poison
#x_poison_all = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])

tr_Y = tr.Y.tondarray() if isinstance(tr.Y, CArray) else tr.Y

#y_poison_all = y_poison
#y_poison_all = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])

# Tüm x ve y değerlerini birleştiriyoruz
all_x_poison = np.concatenate([point[0].tondarray() if isinstance(point[0], CArray) else point[0] for point in poisoning_points])
all_y_poison = np.concatenate([point[1].tondarray() if isinstance(point[1], CArray) else point[1] for point in poisoning_points])

# tr size'ın %20'si kadar bir örnekleme boyutu belirliyoruz
sample_size = int(tr.X.shape[0] * 0.20)

# Rastgele sample_size kadar noktayı seçiyoruz
random_indices = np.random.choice(len(all_x_poison), sample_size, replace=False)
x_poison_all = all_x_poison[random_indices]
y_poison_all = all_y_poison[random_indices]

all_X = np.concatenate([tr_X, x_poison_all])
all_Y = np.concatenate([tr_Y, y_poison_all])

print(len(x_poison_all))
print(len(y_poison_all))

class_label1 = 1
class_indices1 = np.where(all_Y == class_label1)[0]
class_label2 = 0
class_indices2 = np.where(all_Y == class_label2)[0]

print(len(all_Y))
print(len(class_indices1))
print(len(class_indices2))
