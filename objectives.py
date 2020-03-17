import numpy as np

from abc import abstractmethod

eps = 1e-200


class Objective(object):

    def set_model(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    @abstractmethod
    def evaluate(self, w):
        pass


class RSSObjective(Objective):

    def evaluate(self, w):
        y_i = w[0] + np.dot(self.X, w[1:])
        error = np.linalg.norm(self.y - y_i, ord=2)
        return -error


class ScoreObjective(Objective):

    def evaluate(self, w):
        self.model.intercept_ = w[0]
        self.model.coef_ = np.array([w[1:]])
        return self.model.score(self.X, self.y)


class RidgeObjective(Objective):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def evaluate(self, w):
        y_i = w[0] + np.dot(self.X, w[1:])
        penalty = self.alpha * np.linalg.norm(w, ord=2)
        error = np.linalg.norm(self.y - y_i, ord=2) + penalty
        return -error


class CrossEntropyObjective(Objective):

    def set_model(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.model.classes_ = np.unique(y)

    def sigmoid(self, y_i):
        return 1.0 / (1 + np.exp(-y_i))

    def evaluate(self, w):
        y_i = self.sigmoid(w[0] + np.dot(self.X, w[1:]))
        error = np.sum(self.y * np.log(y_i + eps) +
                       (1 - self.y) * np.log((1 - y_i) + eps))
        return error
