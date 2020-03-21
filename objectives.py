import numpy as np

from abc import abstractmethod

eps = 1e-200


class Objective(object):

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def evaluate(self, w):
        pass


class RSSObjective(Objective):

    def evaluate(self, w):
        y_i = w[0] + np.dot(self.model.X_, w[1:])
        error = np.linalg.norm(self.model.y_ - y_i, ord=2)
        return -error


class ScoreObjective(Objective):

    def evaluate(self, w):
        self.model.intercept_ = w[0]
        self.model.coef_ = np.array([w[1:]])
        return self.model.score(self.model.X_, self.model.y_)


class RidgeObjective(Objective):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def evaluate(self, w):
        y_i = w[0] + np.dot(self.model.X_, w[1:])
        penalty = self.alpha * np.linalg.norm(w, ord=2)
        error = np.linalg.norm(self.model.y_ - y_i, ord=2) + penalty
        return -error


class CrossEntropyObjective(Objective):

    def set_model(self, model):
        self.model = model
        self.model.classes_ = np.unique(self.model.y_)

    def sigmoid(self, y_i):
        return 1.0 / (1 + np.exp(-y_i))

    def evaluate(self, w):
        y_i = self.sigmoid(w[0] + np.dot(self.model.X_, w[1:]))
        error = np.sum(self.model.y_ * np.log(y_i + eps) +
                       (1 - self.model.y_) * np.log((1 - y_i) + eps))
        return error
