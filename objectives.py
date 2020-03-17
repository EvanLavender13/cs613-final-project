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


class MultiCrossEntropyObjective(Objective):

    def set_model(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.model.classes_ = np.unique(y)
        self.y_ = np.zeros((y.shape[0], self.model.classes_.shape[0]))
        for i in range(y.shape[0]):
            self.y_[i][np.where(self.model.classes_ == y[i])] = 1

    def softmax(self, y_i):
        e_y = np.exp(y_i - np.max(y_i))
        return e_y / np.sum(e_y)

    def evaluate(self, w):
        print()
        print("classes", self.model.classes_.shape)
        for i, c in enumerate(self.model.classes_):
            print(i, c)
            y_c = self.y[np.where(self.y_[:, i] == 1)]
            print(y_c.shape)
            # y_i = self.softmax(w[0] + np.dot(self.X, w[1:]))
        y_i = w[0] + np.dot(self.X, w[1:])
        print()
        print("y_i", y_i, y_i.shape)
        print("softmax", self.softmax(y_i[0]))
        print("y_", self.y_[:, 0], self.y_[:, 0].shape)
        error = np.sum(self.y_[:, 0] * np.log(y_i + eps))
        print(error)
        1 / 0
        return error
