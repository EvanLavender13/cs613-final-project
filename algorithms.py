from abc import abstractmethod

import numpy as np

from tqdm import tqdm

from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, ClusterMixin
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_random_state


class EvolutionStrategy(BaseEstimator):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        if y is not None:
            X, y = check_X_y(X, y)
        else:
            X = check_array(X)

        self.X_ = X
        self.y_ = y
        self.objective.set_model(self)
        self.sort_ = False

        self.random_ = check_random_state(self.random_state)
        if not self.n_features:
            self.n_features = 1 + X.shape[1]
        self.history_ = np.zeros(self.n_iter)

        self.mu_ = np.ones(self.n_features)
        self.intercept_ = self.mu_[0]
        self.coef_ = np.array([self.mu_[1:]])

        self._pre_iterate()
        for g in tqdm(range(self.n_iter), disable=not self.verbose):
            pop = self._sample_population()
            fit = np.fromiter(map(self.objective.evaluate, pop), np.float)

            if self.sort_:
                self.sort_indices_ = np.argsort(fit)
                fit = fit[self.sort_indices_][::-1]
                pop = pop[self.sort_indices_][::-1]

            self.mu_ = self._update(pop, fit)

            self.intercept_ = self.mu_[0]
            self.coef_ = np.array([self.mu_[1:]])

            self._post_iterate(g)

        return self

    def _pre_iterate(self):
        pass

    def _post_iterate(self, g):
        pass

    @abstractmethod
    def _sample_population(self):
        pass

    @abstractmethod
    def _update(self, pop, fit):
        pass


class SimpleES(EvolutionStrategy):
    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma

    def _pre_iterate(self):
        self.sort_ = True

    def _sample_population(self):
        return self.mu_ + self.sigma * self.random_.randn(self.n_pop, self.n_features)

    def _update(self, pop, fit):
        return pop[0]


class SimpleESRegressor(SimpleES, LinearModel, RegressorMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class SimpleESClassifier(SimpleES, LinearClassifierMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma

    def _pre_iterate(self):
        self.classes_ = np.unique(self.y_)

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class SimpleESClustering(SimpleES, ClusterMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma

    def _post_iterate(self, g):
        n_dim = self.X_[0].shape[0]
        ref_vectors = np.reshape(
            self.mu_, (int(self.n_features / n_dim), n_dim))
        labels = -np.ones(self.X_.shape[0])

        inertia = 0
        for i, sample in enumerate(self.X_):
            distance = np.zeros(ref_vectors.shape[0])
            for j, ref in enumerate(ref_vectors):
                distance[j] = np.linalg.norm(sample - ref) ** 2

            labels[i] = np.argmin(distance)
            inertia += np.min(distance)

        self.labels_ = labels
        self.cluster_centers_ = ref_vectors
        self.history_[g] = inertia
        self.inertia_ = inertia


class GeneticES(EvolutionStrategy):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, p_elite=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.p_elite = p_elite

    def _pre_iterate(self):
        self.sort_ = True
        self.n_elite_ = np.int(self.n_pop * self.p_elite)
        pop = self.mu_ + self.sigma * \
            self.random_.randn(self.n_pop, self.n_features)
        fit = np.fromiter(map(self.objective.evaluate, pop), np.float)
        elite_indices = np.argsort(fit)[-self.n_elite_:]
        self.elite_ = pop[elite_indices]

    def _sample_population(self):
        pop = self.sigma * self.random_.randn(self.n_pop, self.n_features)
        for i in range(self.n_pop):
            index_a = self.random_.choice(self.n_elite_)
            index_b = self.random_.choice(self.n_elite_)

            a = self.elite_[index_a]
            b = self.elite_[index_b]
            c = np.copy(a)
            indices = np.where(self.random_.rand(self.n_features) > 0.5)
            c[indices] = b[indices]
            pop[i] += c

        return pop

    def _update(self, pop, fit):
        self.elite_ = pop[self.n_elite_:]
        return np.mean(self.elite_, axis=0)


class GeneticESRegressor(GeneticES, LinearModel, RegressorMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, p_elite=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.p_elite = p_elite

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class GeneticESClassifier(GeneticES, LinearClassifierMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, p_elite=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.p_elite = p_elite

    def _pre_iterate(self):
        self.classes_ = np.unique(self.y_)

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class GeneticESClustering(GeneticES, ClusterMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, p_elite=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.p_elite = p_elite

    def _post_iterate(self, g):
        n_dim = self.X_[0].shape[0]
        ref_vectors = np.reshape(
            self.mu_, (int(self.n_features / n_dim), n_dim))
        labels = -np.ones(self.X_.shape[0])

        inertia = 0
        for i, sample in enumerate(self.X_):
            distance = np.zeros(ref_vectors.shape[0])
            for j, ref in enumerate(ref_vectors):
                distance[j] = np.linalg.norm(sample - ref) ** 2

            labels[i] = np.argmin(distance)
            inertia += np.min(distance)

        self.labels_ = labels
        self.cluster_centers_ = ref_vectors
        self.history_[g] = inertia
        self.inertia_ = inertia


class NaturalES(EvolutionStrategy):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, alpha=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.alpha = alpha

    def _sample_population(self):
        self.noise_ = self.random_.randn(self.n_pop, self.n_features)
        pop = self.mu_ + self.sigma * self.noise_
        return pop

    def _update(self, pop, fit):
        fit_n = (fit - np.mean(fit)) / np.std(fit, ddof=1)
        return self.mu_ + (self.alpha / (self.n_pop * self.sigma) * np.dot(self.noise_.T, fit_n))


class NaturalESRegressor(NaturalES, LinearModel, RegressorMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, alpha=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.alpha = alpha

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class NaturalESClassifier(NaturalES, LinearClassifierMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, alpha=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.alpha = alpha

    def _pre_iterate(self):
        self.classes_ = np.unique(self.y_)

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class NaturalESClustering(NaturalES, ClusterMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, alpha=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.alpha = alpha

    def _post_iterate(self, g):
        n_dim = self.X_[0].shape[0]
        ref_vectors = np.reshape(
            self.mu_, (int(self.n_features / n_dim), n_dim))
        labels = -np.ones(self.X_.shape[0])

        inertia = 0
        for i, sample in enumerate(self.X_):
            distance = np.zeros(ref_vectors.shape[0])
            for j, ref in enumerate(ref_vectors):
                distance[j] = np.linalg.norm(sample - ref) ** 2

            labels[i] = np.argmin(distance)
            inertia += np.min(distance)

        self.labels_ = labels
        self.cluster_centers_ = ref_vectors
        self.history_[g] = inertia
        self.inertia_ = inertia


class CMAES(EvolutionStrategy):
    # Does not work :(

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma

    def _pre_iterate(self):
        self.sort_ = True
        self.C_ = np.identity(self.n_features)

        self.ps_ = np.zeros((self.n_features, 1))
        self.pc_ = np.zeros((self.n_features, 1))

        self.u_ = self.n_pop // 4
        self.w_ = np.asarray(range(1, self.u_ + 1))[::-1]
        self.w_ = self.w_ / np.sum(self.w_)

        self.cc_ = 4 / self.n_features
        self.cs_ = 4 / self.n_features
        self.cu_ = self.u_ / (self.n_features ** 2)
        self.c1_ = 1 - self.cu_

        self.ds_ = 1 + np.sqrt(self.u_ / self.n_features)
        self.exp_ = np.sqrt(self.n_features) * \
            (1 - 1 / (4 * self.n_features) + 1 / (21 * (self.n_features ** 2)))

    def _sample_population(self):
        self.yi_ = self.random_.multivariate_normal(
            np.zeros(self.n_features), self.C_, size=self.n_pop)
        pop = self.mu_ + self.sigma * self.yi_
        return pop

    def _update(self, pop, fit):
        # yi = self.yi_[self.sort_indices_][::-1][:self.u_]
        # print("yi\n", yi)
        # yw = np.dot(self.w_, yi)
        # print("yw\n", yw)
        # new_mu = self.mu_ + self.sigma * yw
        new_mu = self.mu_ + np.dot(self.w_, pop[:self.u_] - self.mu_)
        # print("new_mu\n", new_mu)

        mu_displacement = (new_mu - self.mu_) / self.sigma
        norm = np.linalg.norm(self.ps_, ord=1)
        indicator = 1 if norm < (1.5 * np.sqrt(self.n_features)) else 0
        self.pc_ = (1 - self.cc_) * self.pc_ + indicator * \
            np.sqrt(1 - (1 - self.cc_) ** 2) * \
            np.sqrt(self.u_) * mu_displacement
        # print("self.pc_\n", self.pc_)

        C_sqrt = sqrtm(np.linalg.inv(self.C_))
        self.ps_ = (1 - self.cs_) * self.ps_ + \
            np.sqrt(1 - (1 - self.cs_) ** 2) * \
            np.sqrt(self.u_) * np.dot(C_sqrt, mu_displacement)
        # print("self.ps_\n", self.ps_)

        rank_one = self.c1_ * np.dot(self.pc_, self.pc_.T)
        # print("rank_one\n", rank_one)

        diff = (pop[:self.u_] - self.mu_) / self.sigma
        w = np.diag(self.w_)
        rank_min = self.cu_ * np.dot(np.dot(diff.T, w), diff)
        # print("rank_min\n", rank_min)

        self.C_ = (1 - self.c1_ - self.cu_) * self.C_ + rank_one + rank_min
        # print("self.C_\n", self.C_)

        damp = self.cs_ / self.ds_
        norm = np.linalg.norm(self.ps_, ord=1)
        ratio = norm / self.exp_ - 1
        self.sigma = self.sigma * np.exp(damp * ratio)
        print("self.sigma", self.sigma)

        return new_mu


class CMAESRegressor(CMAES, LinearModel, RegressorMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, verbose=True, sigma=1.0, p_elite=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma
        self.p_elite = p_elite

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class DifferentialEvolution(BaseEstimator):

    def __init__(self, objective=None, n_features=None, n_pop=10, n_iter=100, random_state=None, verbose=True, bounds=None, F=0.8, cx_pb=0.7, lmbda=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.bounds = bounds
        self.F = F
        self.cx_pb = cx_pb
        self.lmbda = lmbda

    def fit(self, X, y=None):
        if y is not None:
            X, y = check_X_y(X, y)
        else:
            X = check_array(X)

        self.X_ = X
        self.y_ = y
        self.objective.set_model(self)

        self.random_ = check_random_state(self.random_state)
        if not self.n_features:
            self.n_features = 1 + X.shape[1]
        self.history_ = np.zeros(self.n_iter)

        self.bounds_ = np.array([self.bounds] * self.n_features)
        self.mu_ = np.ones(self.n_features)
        self.intercept_ = self.mu_[0]
        self.coef_ = np.array([self.mu_[1:]])

        min_bound, max_bound = self.bounds_[:, 0], self.bounds_[:, 1]
        pop = self.random_.uniform(
            min_bound, max_bound, (self.n_pop, self.n_features))
        fit = np.fromiter(map(self.objective.evaluate, pop), np.float)
        best_index = np.argmax(fit)

        for g in tqdm(range(self.n_iter), disable=not self.verbose):

            for i in range(self.n_pop):
                # choose random vectors
                r_i = [j for j in range(self.n_pop) if j != i]
                # x1, x2, x3 = pop[self.random_.choice(r_i, 3, replace=False)]

                x_i = pop[i]
                x_best = pop[best_index]
                x2, x3 = pop[self.random_.choice(r_i, 2, replace=False)]

                # create trial vector

                # scheme DE1
                # trial = np.clip(x1 + self.F * (x2 - x3), min_bound, max_bound)
                # trial = x1 + self.F * (x2 - x3)

                # scheme DE2
                v = x_i + self.lmbda * \
                    (x_best - x_i) + self.F * (x2 - x3)

                # determine crossover points
                # method from paper
                # n = self.random_.randint(0, self.n_features - 1)
                # L = 1

                # while self.random_.rand() < self.cx_pb and L < self.n_features:
                #    L += 1

                # v = np.arange(n, n + L - 1) % self.n_features
                # u = np.copy(pop[i])
                # u[v] = trial[v]

                cx_points = self.random_.rand(self.n_features) < self.cx_pb
                if not np.any(cx_points):
                    cx_points[self.random_.randint(0, self.n_features)] = True

                u = pop[i]
                trial = np.where(cx_points, v, u)

                f = self.objective.evaluate(trial)
                if f > fit[i]:
                    fit[i] = f
                    pop[i] = trial

                    if f > fit[best_index]:
                        best_index = i

            self.mu_ = pop[best_index]
            self.intercept_ = self.mu_[0]
            self.coef_ = np.array([self.mu_[1:]])

            self._post_iterate(g)

        return self

    def _post_iterate(self, g):
        pass


class DifferentialEvolutionRegressor(DifferentialEvolution, LinearModel, RegressorMixin):

    def __init__(self, objective=None, n_features=None, n_pop=10, n_iter=100, random_state=None, verbose=True, bounds=None, F=0.8, cx_pb=0.7, lmbda=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.bounds = bounds
        self.F = F
        self.cx_pb = cx_pb
        self.lmbda = lmbda

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class DifferentialEvolutionClassifier(DifferentialEvolution, LinearClassifierMixin):

    def __init__(self, objective=None, n_features=None, n_pop=10, n_iter=100, random_state=None, verbose=True, bounds=None, F=0.8, cx_pb=0.7, lmbda=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.bounds = bounds
        self.F = F
        self.cx_pb = cx_pb
        self.lmbda = lmbda

    def _pre_iterate(self):
        self.classes_ = np.unique(self.y_)

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class DifferentialEvolutionClustering(DifferentialEvolution, ClusterMixin):

    def __init__(self, objective=None, n_features=None, n_pop=10, n_iter=100, random_state=None, verbose=True, bounds=None, F=0.8, cx_pb=0.7, lmbda=1.0):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.bounds = bounds
        self.F = F
        self.cx_pb = cx_pb
        self.lmbda = lmbda

    def _post_iterate(self, g):
        n_dim = self.X_[0].shape[0]
        ref_vectors = np.reshape(
            self.mu_, (int(self.n_features / n_dim), n_dim))
        labels = -np.ones(self.X_.shape[0])

        inertia = 0
        for i, sample in enumerate(self.X_):
            distance = np.zeros(ref_vectors.shape[0])
            for j, ref in enumerate(ref_vectors):
                distance[j] = np.linalg.norm(sample - ref) ** 2

            labels[i] = np.argmin(distance)
            inertia += np.min(distance)

        self.labels_ = labels
        self.cluster_centers_ = ref_vectors
        self.history_[g] = inertia
        self.inertia_ = inertia
