import sys

sys.path.extend(["./"])

from data.mnist_loader import *
from src.experiments.run_attack_all import *
from src.experiments.run_attack_2 import *
from src.classifier.secml_classifier import SVMClassifier, LogisticClassifier, RidgeClassifier, DecisionTreeClassifierWrapper
from src.classifier.secml_classifier_2 import LDAClassifier, NaiveBayesClassifier
from src.optimizer.beta_optimizer import beta_poison, to_scaled_img
from src.optimizer.flip_poisoning import flip_batch_poison
from src.optimizer.white_poisoning import white_poison
import os

if __name__ == "__main__":
    set_seed(444)
    d1, d2 = int(opts.ds[0]), int(opts.ds[2])
    digits = (d1, d2)
    tr, val, ts = load_mnist(digits=digits, n_tr=400, n_val=1000, n_ts=1000)

    if opts.classifier == "logistic":
        clf = LogisticClassifier()
    elif opts.classifier == "ridge":
        clf = RidgeClassifier(alpha=0.5)
    elif opts.classifier == "decision_tree":
        clf = DecisionTreeClassifierWrapper(max_depth=3)
    elif opts.classifier == "lda":
        clf = LDAClassifier()
    elif opts.classifier == "naive_bayes":
        clf = NaiveBayesClassifier()
    else:
        clf = SVMClassifier(k="linear")
    params = {
        "n_proto": opts.n_proto,
        "lb": 1,
        "y_target": None,
        "y_poison": None,
        "transform": to_scaled_img,
    }
    path = opts.path + "/mnist-{}-tr{}/{}/".format(
        opts.ds, tr.X.shape[0], opts.classifier
    )
    os.makedirs(path, exist_ok=True)

    if "ridge" in opts.classifier:
        name = path + "beta_poison_ridge" + str(opts.n_proto)
        run_attack(beta_poison, name, clf, tr, val, ts, params=params)
    if "svm" in opts.classifier:
        name = path + "beta_poison_svm" + str(opts.n_proto)
        run_attack(beta_poison, name, clf, tr, val, ts, params=params)
    if "logistic" in opts.classifier:
        name = path + "beta_poison_logistic" + str(opts.n_proto)
        run_attack(beta_poison, name, clf, tr, val, ts, params=params)
    if "decision_tree" in opts.classifier:
        name = path + "beta_poison_decision_tree" + str(opts.n_proto)
        run_attack(beta_poison, name, clf, tr, val, ts, params=params)
    if "lda" in opts.classifier:
        name = path + "beta_poison_lda" + str(opts.n_proto)
        run_attack_2(beta_poison, name, clf, tr, val, ts, params=params)
    if "naive_bayes" in opts.classifier:
        name = path + "beta_poison_naive_bayes" + str(opts.n_proto)
        run_attack_2(beta_poison, name, clf, tr, val, ts, params=params)
