from sklearn import svm
from sklearn.externals import joblib


class LinearSVC(svm.LinearSVC):
    def save(self, path):
        joblib.dump(self, path)
