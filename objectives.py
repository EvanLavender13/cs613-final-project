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


class CrossEntropyObjective(Objective):

    def set_model(self, model):
        self.model = model

    def sigmoid(self, y_i):
        return 1.0 / (1 + np.exp(-y_i))

    def evaluate(self, w):
        y_i = self.sigmoid(w[0] + np.dot(self.model.X_, w[1:]))
        error = np.sum(self.model.y_ * np.log(y_i + eps) +
                       (1 - self.model.y_) * np.log((1 - y_i) + eps))
        return error


class KMeansObjective(Objective):

    def evaluate(self, w):
        n_dim = self.model.X_[0].shape[0]
        n_features = self.model.n_features
        ref_vectors = np.reshape(w, (int(n_features / n_dim), n_dim))

        total_distance = 0
        for sample in self.model.X_:
            distance = np.zeros(ref_vectors.shape[0])
            for i, ref in enumerate(ref_vectors):
                distance[i] = np.linalg.norm(sample - ref) ** 2

            total_distance += np.min(distance)

        return -total_distance
