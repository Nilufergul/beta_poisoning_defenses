from secml.ml.classifiers import CClassifierSVM, CClassifierLogistic, CClassifierRidge, CClassifierDecisionTree
from secml.ml.classifiers.loss import CLossCrossEntropy
from src.classifier.secml_autograd import SecmlLayer, as_tensor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def to_tensor(x, device="cpu"):
    x = as_tensor(x).unsqueeze(0)
    x = x.float()
    x = x.to(device)
    return x


class SecmlClassifier:
    def __init__(self, clf=None, preprocess=None, n_classes=2):
        self.clf = clf
        self.fitted = False
        self.n_labels = n_classes
        self.C = self.alpha = None
        self._preprocess = preprocess
        self.device = "cpu"

        if self._preprocess is None:
            self._preprocess = to_tensor
        self.device = "cpu"

    def loss(self, x, labels):
        scores = self.clf.decision_function(x)
        loss = self.clf._loss.loss(labels, scores).sum()
        return loss

    def ce_loss(self, x, labels):
        scores = self.clf.decision_function(x)
        prob = scores  # scores.exp() / (scores.exp()).sum(axis=1)
        ce = CLossCrossEntropy()
        return ce.loss(y_true=labels, score=prob).sum()

    def predict(self, x, return_decision_function=False):
        if not return_decision_function:
            predicted = self.clf.predict(x)
            return predicted
        else:
            predicted, scores = self.clf.predict(x, return_decision_function=True)
            return predicted, scores

    def _fit(self, ds):
        return self.fit(ds)

    def _forward(self, x):
        return self.predict(x)

    def fit(self, ds):
        self.clf.fit(ds.X, ds.Y)

    def init(self):
        pass

    def init_fit(self, ds, parameters):
        svm = CClassifierSVM(C=parameters["C"], store_dual_vars=True)
        self.clf = svm
        self.fit(ds.X, ds.Y)

    def preprocess(self, x):
        return self._preprocess(x)

    def deepcopy(self):
        pass

    def to_string(self):
        pass

    def is_fitted(self):
        return self.fitted

    @property
    def n_classes(self):
        """Number of classes of training dataset."""
        return self.n_labels

    def transform(self, x):
        return x

    def transform_all(self, x):
        return x

    def _to_torch_classifier(self):
        return SecmlLayer(self.clf)


class SVMClassifier(SecmlClassifier):
    __class_type = "svm"

    def __init__(self, clf=None, k=None):
        super().__init__(clf)
        if clf is not None:
            self.C = clf.C
            self.kernel = clf.kernel
        else:
            # init with params
            self.kernel = k

    def init_fit(self, ds, parameters):
        svm = CClassifierSVM(C=parameters["C"], kernel=self.kernel)
        self.clf = svm
        self.fit(ds)
        self.fitted = True
        self.C = parameters["C"]

    def loss(self, x, labels):
        return self.ce_loss(x, labels)

    def deepcopy(self):
        return SVMClassifier(self.clf.deepcopy())

    def to_string(self):
        return "svm"


class LogisticClassifier(SecmlClassifier):
    __class_type = "logistic"

    def __init__(self, clf=None):
        super().__init__(clf)

    def init_fit(self, ds, parameters):
        svm = CClassifierLogistic(C=parameters["C"])
        self.clf = svm
        self.fit(ds)
        self.fitted = True

    def deepcopy(self):
        return LogisticClassifier(self.clf.deepcopy())

    def to_string(self):
        return "logistic"

    def loss(self, x, labels):
        return self.ce_loss(x, labels)


class RidgeClassifier(SecmlClassifier):
    __class_type = "ridge"

    def __init__(self, clf=None, alpha=1.0):
        super().__init__(clf)
        self.alpha = alpha

    def init_fit(self, ds, parameters=None):
        if parameters and "alpha" in parameters:
            self.alpha = parameters["alpha"]
        ridge = CClassifierRidge(alpha=self.alpha)
        self.clf = ridge
        self.fit(ds)
        self.fitted = True

    def deepcopy(self):
        return RidgeClassifier(self.clf.deepcopy(), self.alpha)

    def to_string(self):
        return "ridge"

    def loss(self, x, labels):
        return self.ce_loss(x, labels)
    
class DecisionTreeClassifierWrapper(SecmlClassifier):
    __class_type = "decision_tree"

    def __init__(self, clf=None, max_depth=None):
        super().__init__(clf)
        self.max_depth = max_depth

    def init_fit(self, ds, parameters=None):
        if parameters and "max_depth" in parameters:
            self.max_depth = parameters["max_depth"]
        self.clf = CClassifierDecisionTree(criterion='gini', max_depth=self.max_depth)
        self.fit(ds)
        self.fitted = True

    def fit(self, ds):
        self.clf.fit(ds.X, ds.Y)

    def predict(self, x, return_decision_function=False):
        if not return_decision_function:
            predicted = self.clf.predict(x)
            return predicted
        else:
            predicted = self.clf.predict(x)
            scores = self.clf.predict_proba(x)
            return predicted, scores

    def deepcopy(self):
        return DecisionTreeClassifierWrapper(self.clf)

    def to_string(self):
        return "decision_tree"

    def loss(self, x, labels):
        # DecisionTreeClassifier doesn't have a decision_function, use predict_proba
        scores = self.clf.predict_proba(x)
        ce = CLossCrossEntropy()
        return ce.loss(y_true=labels, score=scores).sum()
    
class LDAClassifier(SecmlClassifier):
    __class_type = "lda"

    def __init__(self, clf=None):
        super().__init__(clf)

    def init_fit(self, ds, parameters=None):
        self.clf = LinearDiscriminantAnalysis()
        self.fit(ds)
        self.fitted = True

    def fit(self, ds):
        self.clf.fit(ds.X, ds.Y)

    def predict(self, x, return_decision_function=False):
        if not return_decision_function:
            predicted = self.clf.predict(x)
            return predicted
        else:
            predicted = self.clf.predict(x)
            scores = self.clf.predict_proba(x)
            return predicted, scores

    def deepcopy(self):
        return LDAClassifier(self.clf)

    def to_string(self):
        return "lda"

    def loss(self, x, labels):
        return self.ce_loss(x, labels)