"""Feature selection methods from the paper: RF Gini, SHAP, stepwise LDA, and DISR."""
import numpy
import pandas
import shap
from scipy.stats import f as f_distribution
from sklearn.ensemble import RandomForestClassifier


def random_forest_rankings(X_train, y_train, X_explain, shap_samples=100, seed=0):
    """Fit a random forest and return it with features ranked by Gini importance and mean |SHAP|."""
    forest = RandomForestClassifier(random_state=seed, n_jobs=-1).fit(X_train, y_train)
    columns = numpy.array(X_train.columns)
    gini = columns[numpy.argsort(forest.feature_importances_)[::-1]].tolist()

    values = shap.TreeExplainer(forest).shap_values(X_explain.iloc[:shap_samples])
    # SHAP returns a list per class or an (samples, features, classes) array, depending on its version.
    values = numpy.abs(numpy.stack(values, axis=-1) if isinstance(values, list) else numpy.asarray(values))
    importance = values.mean(axis=(0, 2)) if values.ndim == 3 else values.mean(axis=0)
    shapley = columns[numpy.argsort(importance)[::-1]].tolist()

    return forest, gini, shapley


def stepwise_lda(X, y, p_enter=0.1):
    """Return features in forward selection order, using the partial Wilks' lambda F test.

    Each step adds the feature with the smallest p-value, and selection stops when none
    is below `p_enter`.
    """
    values = X.to_numpy(dtype=float)
    labels = pandas.factorize(y)[0]
    n_samples, n_groups = len(values), labels.max() + 1

    total = numpy.cov(values, rowvar=False, bias=True) * n_samples
    within = numpy.zeros_like(total)
    for label in range(n_groups):
        members = values[labels == label]
        within += numpy.cov(members, rowvar=False, bias=True) * len(members)

    selected = []
    remaining = list(range(values.shape[1]))
    while remaining:
        candidates = numpy.array(remaining)
        lambdas = _partial_lambda(within, total, selected, candidates)
        usable = ~numpy.isnan(lambdas)
        degrees = n_samples - n_groups - len(selected)
        if not usable.any() or degrees <= 0:
            break

        statistic = (1 - lambdas[usable]) / lambdas[usable] * degrees / (n_groups - 1)
        p_values = f_distribution.sf(statistic, n_groups - 1, degrees)
        best = numpy.argmin(p_values)
        if p_values[best] >= p_enter:
            break

        feature = int(candidates[usable][best])
        selected.append(feature)
        remaining.remove(feature)

    return [X.columns[index] for index in selected]


def _partial_lambda(within, total, selected, candidates):
    """Return each candidate's Wilks' lambda given the selected features; NaN if it adds no variance."""
    def residual(matrix):
        diagonal = matrix[candidates, candidates]
        if not selected:
            return diagonal
        cross = matrix[numpy.ix_(selected, candidates)]
        solved = numpy.linalg.solve(matrix[numpy.ix_(selected, selected)], cross)
        return diagonal - (cross * solved).sum(axis=0)

    numerator, denominator = residual(within), residual(total)
    tolerance = 1e-10 * numpy.maximum(total[candidates, candidates], 1e-300)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        return numpy.where((denominator > tolerance) & (numerator > tolerance), numerator / denominator, numpy.nan)


def discretize(X, bins=6):
    """Bin each column into quantiles, returning integer codes; a constant column becomes one bin."""
    return X.apply(lambda column: pandas.qcut(column, bins, labels=False, duplicates="drop")).fillna(0).astype(int)


def disr(X, y, k=3, bins=6):
    """Return `k` features by Double Input Symmetrical Relevance (Meyer and Bontempi, 2006).

    Starts from the feature with the highest mutual information with `y`, then greedily
    adds the feature maximizing the summed I(Xi,Xs;Y) / H(Xi,Xs,Y) over the selected Xs.
    """
    codes = discretize(X, bins).to_numpy()
    labels = pandas.factorize(y)[0]
    candidates = list(range(codes.shape[1]))

    relevance = [_mutual_information(codes[:, index], labels) for index in candidates]
    selected = [candidates.pop(int(numpy.argmax(relevance)))]

    scores = numpy.zeros(codes.shape[1])
    while candidates and len(selected) < k:
        latest = codes[:, selected[-1]]
        for index in candidates:
            scores[index] += _symmetrical_relevance(codes[:, index], latest, labels)
        best = max(candidates, key=lambda index: scores[index])
        candidates.remove(best)
        selected.append(best)

    return [X.columns[index] for index in selected]


def _combine(first, second):
    return first * (second.max() + 1) + second


def _entropy(codes):
    counts = numpy.unique(codes, return_counts=True)[1]
    probabilities = counts / counts.sum()
    return float(-(probabilities * numpy.log2(probabilities)).sum())


def _mutual_information(first, second):
    return _entropy(first) + _entropy(second) - _entropy(_combine(first, second))


def _symmetrical_relevance(first, second, labels):
    pair = _combine(first, second)
    joint = _entropy(_combine(pair, labels))
    return (_entropy(pair) + _entropy(labels) - joint) / joint if joint > 0 else 0.0
