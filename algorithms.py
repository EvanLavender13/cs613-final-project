from abc import abstractmethod

import numpy as np

from tqdm import tqdm

from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_random_state


class EvolutionStrategy(BaseEstimator):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.classifier = False

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.objective.set_model(self, X, y)
        self.X_ = X
        self.y_ = y

        self.random_ = check_random_state(self.random_state)
        if not self.n_features:
            self.n_features = 1 + X.shape[1]
        self.history_ = np.zeros(self.n_iter)

        self.mu_ = np.zeros(self.n_features)
        self.intercept_ = self.mu_[0]
        self.coef_ = np.array([self.mu_[1:]])

        if self.classifier:
            self.classes_ = np.unique(y)

        self._pre_iterate()
        for g in tqdm(range(self.n_iter)):
            pop = self._sample_population()
            fit = np.fromiter(map(self.objective.evaluate, pop), np.float)
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
    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.sigma = sigma

    def _sample_population(self):
        return self.mu_ + self.sigma * self.random_.randn(self.n_pop, self.n_features)

    def _update(self, pop, fit):
        best = pop[np.argmax(fit)]
        return best


class SimpleESRegressor(SimpleES, LinearModel, RegressorMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.sigma = sigma
        self.classifier = False

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class SimpleESClassifier(SimpleES, LinearClassifierMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.sigma = sigma
        self.classifier = True

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class GeneticES(EvolutionStrategy):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2, p_elite=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.sigma = sigma
        self.p_elite = p_elite
        self.classifier = True

    def _pre_iterate(self):
        self.n_elite_ = np.int(self.n_pop * self.p_elite)
        pop = self.mu_ + self.sigma * \
            self.random_.randn(self.n_pop, self.n_features)
        fit = np.fromiter(map(self.objective.evaluate, pop), np.float)
        elite_indices = np.argsort(fit)[-np.int(self.n_elite_):]
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
        elite_indices = np.argsort(fit)[-np.int(self.n_elite_):]
        self.elite_ = pop[elite_indices]

        return np.mean(self.elite_, axis=0)


class GeneticESRegressor(GeneticES, LinearModel, RegressorMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2, p_elite=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.sigma = sigma
        self.p_elite = p_elite
        self.classifier = False

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class GeneticESClassifier(GeneticES, LinearClassifierMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2, p_elite=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.sigma = sigma
        self.p_elite = p_elite
        self.classifier = True

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class NaturalES(EvolutionStrategy):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2, alpha=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
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

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2, alpha=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.sigma = sigma
        self.alpha = alpha
        self.classifier = False

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class NaturalESClassifier(NaturalES, LinearClassifierMixin):

    def __init__(self, objective=None, n_features=None, n_pop=100, n_iter=100, random_state=None, sigma=0.2, alpha=0.1):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.sigma = sigma
        self.alpha = alpha
        self.classifier = True

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class CMAES(BaseEstimator, RegressorMixin):

    def __init__(self, objective=None, n_features=None, pop_size=100, iterations=100, sigma=0.2, num_elite=10):
        self.objective = objective
        self.n_features = n_features
        self.pop_size = pop_size
        self.iterations = iterations
        self.sigma = sigma
        self.num_elite = num_elite

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.n_ = X.shape[1]
        self.mu_ = np.zeros(self.n_)

        self.history_ = np.zeros(self.iterations)

        self.cov_ = np.identity(self.n_)
        self.mvn_ = multivariate_normal(mean=self.mu_, cov=self.cov_)

        zero = np.zeros(self.n_)
        for g in range(self.iterations):
            # sample
            pop = self.mu_ + self.sigma * \
                np.random.multivariate_normal(zero, self.cov_, self.pop_size)

            # evaluate
            fit = np.fromiter(map(self.objective.evaluate, pop), np.float)

            # update
            elite_indices = np.argsort(fit)[-np.int(self.num_elite):]
            elite_pop = pop[elite_indices]
            # print(elite_pop)
            old_mu = self.mu_
            self.mu_ = np.mean(elite_pop, axis=0)
            # print("mu", self.mu_)

            elite_pop -= old_mu
            self.cov_ = np.dot(elite_pop.T, elite_pop.conj()
                               ) / (self.num_elite - 1)
            # print(self.cov_)
            # print()

            best_index = np.argmax(fit)
            self.history_[g] = fit[best_index]

        self.theta_ = self.mu_

        return self

    def predict(self, X):
        X = check_array(X)

        return np.dot(X, self.theta_)


class DifferentialEvolution(BaseEstimator):

    def __init__(self, objective=None, n_features=None, n_pop=10, n_iter=100, random_state=None, bounds=None, F=0.8, cx_pb=0.6):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.bounds = bounds
        self.F = F
        self.cx_pb = cx_pb
        self.classifier = False

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.objective.set_model(self, X, y)
        self.X_ = X
        self.y_ = y

        self.random_ = check_random_state(self.random_state)
        if not self.n_features:
            self.n_features = 1 + X.shape[1]
        self.history_ = np.zeros(self.n_iter)

        self.mu_ = np.zeros(self.n_features)
        self.intercept_ = self.mu_[0]
        self.coef_ = np.array([self.mu_[1:]])

        if self.classifier:
            self.classes_ = np.unique(y)

        min_bound, max_bound = self.bounds.T
        pop = self.random_.uniform(
            min_bound, max_bound, (self.n_pop, self.n_features))
        fit = np.fromiter(map(self.objective.evaluate, pop), np.float)

        for g in tqdm(range(self.n_iter)):

            for i in range(self.n_pop):
                # choose random vectors
                r_i = [j for j in range(self.n_pop) if j != i]
                x1, x2, x3 = pop[self.random_.choice(r_i, 3, replace=False)]

                # create trial vector
                trial = x1 + self.F * (x2 - x3)

                # determine crossover points
                n = self.random_.randint(0, self.n_features - 1)
                L = 1

                while self.random_.rand() < self.cx_pb and L < self.n_features:
                    L += 1

                v = np.arange(n, n + L - 1) % self.n_features
                u = np.copy(pop[i])
                u[v] = trial[v]

                f = self.objective.evaluate(u)
                if f > fit[i]:
                    fit[i] = f
                    pop[i] = u

            self._post_iterate(g)

            best_index = np.argmax(fit)

            self.mu_ = pop[best_index]
            self.intercept_ = self.mu_[0]
            self.coef_ = np.array([self.mu_[1:]])

        return self

    def _post_iterate(self, g):
        pass


class DifferentialEvolutionRegressor(DifferentialEvolution, LinearModel, RegressorMixin):

    def __init__(self, objective=None, n_features=None, n_pop=10, n_iter=100, random_state=None, bounds=None, F=0.8, cx_pb=0.6):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.bounds = bounds
        self.F = F
        self.cx_pb = cx_pb
        self.classifier = False

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)


class DifferentialEvolutionClassifier(DifferentialEvolution, LinearClassifierMixin):

    def __init__(self, objective=None, n_features=None, n_pop=10, n_iter=100, random_state=None, bounds=None, F=0.8, cx_pb=0.6):
        self.objective = objective
        self.n_features = n_features
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.bounds = bounds
        self.F = F
        self.cx_pb = cx_pb
        self.classifier = True

    def _post_iterate(self, g):
        self.history_[g] = self.score(self.X_, self.y_)
