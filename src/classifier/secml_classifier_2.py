from secml.ml.classifiers.loss import CLossCrossEntropy
from src.classifier.secml_autograd import SecmlLayer, as_tensor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

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
        prob = scores
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
        X = ds.X.tondarray() if hasattr(ds.X, 'tondarray') else ds.X
        y = ds.Y.tondarray().ravel() if hasattr(ds.Y, 'tondarray') else ds.Y.ravel()
        self.clf.fit(X, y)

    def init(self):
        pass

    def init_fit(self, ds, parameters):
        self.clf = LinearDiscriminantAnalysis()
        self.fit(ds)

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

class LDAClassifier(SecmlClassifier):
    __class_type = "lda"

    def __init__(self, clf=None):
        super().__init__(clf)

    def init_fit(self, ds, parameters=None):
        self.clf = LinearDiscriminantAnalysis()
        self.fit(ds)
        self.fitted = True

    def fit(self, ds):
        X = ds.X.tondarray() if hasattr(ds.X, 'tondarray') else ds.X
        y = ds.Y.tondarray().ravel() if hasattr(ds.Y, 'tondarray') else ds.Y.ravel()
        self.clf.fit(X, y)

    def predict(self, x, return_decision_function=False):
        X = x.tondarray() if hasattr(x, 'tondarray') else x
        if not return_decision_function:
            predicted = self.clf.predict(X)
            return predicted
        else:
            predicted = self.clf.predict(X)
            scores = self.clf.predict_proba(X)
            return predicted, scores

    def deepcopy(self):
        return LDAClassifier(self.clf)

    def to_string(self):
        return "lda"

    def loss(self, x, labels):
        X = x.tondarray() if hasattr(x, 'tondarray') else x
        scores = self.clf.predict_proba(X)
        ce = CLossCrossEntropy()
        return ce.loss(y_true=labels, score=scores).sum()
    
class NaiveBayesClassifier(SecmlClassifier):
    __class_type = "naive_bayes"

    def __init__(self, clf=None):
        super().__init__(clf)

    def init_fit(self, ds, parameters=None):
        self.clf = GaussianNB()
        self.fit(ds)
        self.fitted = True

    def fit(self, ds):
        X = ds.X.tondarray() if hasattr(ds.X, 'tondarray') else ds.X
        y = ds.Y.tondarray().ravel() if hasattr(ds.Y, 'tondarray') else ds.Y.ravel()
        self.clf.fit(X, y)

    def predict(self, x, return_decision_function=False):
        X = x.tondarray() if hasattr(x, 'tondarray') else x
        if not return_decision_function:
            predicted = self.clf.predict(X)
            return predicted
        else:
            predicted = self.clf.predict(X)
            scores = self.clf.predict_proba(X)
            return predicted, scores

    def deepcopy(self):
        return NaiveBayesClassifier(self.clf)

    def to_string(self):
        return "naive_bayes"

    def loss(self, x, labels):
        X = x.tondarray() if hasattr(x, 'tondarray') else x
        scores = self.clf.predict_proba(X)
        ce = CLossCrossEntropy()
        return ce.loss(y_true=labels, score=scores).sum()