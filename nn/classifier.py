from sklearn import svm
from sklearn.externals import joblib


class LinearSVC(object):
    def __init__(self):
        self.clsfr = svm.LinearSVC()

    def fit(self, x, y):
        return self.clsfr.fit(x, y)

    def predict(self, x):
        return self.clsfr.predict(x)

    def save(self, path):
        joblib.dump(self.clsfr, path)
