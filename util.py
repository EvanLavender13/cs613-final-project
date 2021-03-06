import matplotlib.pyplot as plt

from sklearn.metrics import plot_precision_recall_curve, classification_report
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def print_scores(model, X, y):
    name = type(model).__name__
    y_pred = model.predict(X)
    print(name)
    print("R2 Score:                 %s" % r2_score(y, y_pred))
    print("Explained Variance Score: %s" % explained_variance_score(y, y_pred))
    print("RMSE:                     %s" %
          mean_squared_error(y, y_pred, squared=False))


def plot_lr_class(model, X, y, ax):
    y_pred = model.predict(X)
    print("LogisticRegression")
    print(classification_report(y, y_pred))
    plot_precision_recall_curve(model, X, y, ax=ax)


def plot_cluster(model, X, y, axes):
    name = type(model).__name__
    print(name)
    print("Homogeneity:  %s\nCompleteness: %s\nV-Measure:    %s" %
          homogeneity_completeness_v_measure(y, model.labels_))
    print("Inertia:      %s" % model.inertia_)
    axes[0].set_title("Clusters")
    axes[0].scatter(model.cluster_centers_[:, 0], model.cluster_centers_[
        :, 1], marker="x", color="black", s=169, linewidths=3, zorder=10, label="Centroid")
    axes[0].scatter(X[:, 0], X[:, 1], c=model.labels_)
    axes[0].legend()

    if hasattr(model, "history_"):
        axes[1].set_title(name + " Inertia")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Inertia")
        axes[1].plot(range(model.history_.shape[0]), model.history_)


def plot_regress(model, X, y, ax):
    print_scores(model, X, y)

    name = type(model).__name__
    ax.set_title(name + " Score")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("R2 Score")
    ax.plot(range(model.history_.shape[0]), model.history_)


def plot_class(model, X, y, axes):
    name = type(model).__name__
    y_pred = model.predict(X)
    print(name)
    print(classification_report(y, y_pred))
    axes[0].set_title(name + " Score")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Score")
    # axes[0].set_ylim([0, 1])
    axes[0].plot(range(model.history_.shape[0]), model.history_)
    axes[1].set_title("PR Curve")
    plot_precision_recall_curve(model, X, y, ax=axes[1])


def cross_val(model, X, y):
    cv = make_pipeline(StandardScaler(), model)
    return cross_val_score(cv, X, y)
