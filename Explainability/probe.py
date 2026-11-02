"""Probe embeddings for a task (paper Section IV) and rank the dimensions most relevant to it.

Folds never share a group, so `group` should be the speaker for gender and the session
for emotion. Prints the cross-validated score, and writes one row per dimension and
method to `<output>.csv` and the parallel categories figure to `<output>.html`.
"""
import argparse

import numpy
import pandas
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import load_embeddings, load_labels, write_parallel_figure
from selection import random_forest_rankings, stepwise_lda


# Accuracy for gender; unweighted average recall, which is balanced accuracy, for emotion.
TASKS = {
        "gender": (RandomForestClassifier(random_state=0, n_jobs=-1), accuracy_score),
        "emotion": (make_pipeline(StandardScaler(), SVC()), balanced_accuracy_score),
        }


def make_folds(X, y, groups, folds):
    if groups.nunique() <= folds:
        return list(LeaveOneGroupOut().split(X, y, groups))
    return list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0).split(X, y, groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", help="pickle written by SPINE.EmbeddingWriter")
    parser.add_argument("labels", help="CSV with utterance_id, group, and label columns")
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("output", help="output path, without extension")
    parser.add_argument("--binarize", action="store_true", help="threshold embeddings before probing")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top", type=int, default=5, help="dimensions kept per method")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    X = load_embeddings(args.embeddings).reindex(labels.index).dropna()
    if args.binarize:
        X = (X >= args.threshold).astype(int)
    y, groups = labels.loc[X.index, "label"], labels.loc[X.index, "group"]

    model, metric = TASKS[args.task]
    folds = make_folds(X, y, groups, args.folds)
    scores = []
    for train, test in folds:
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        scores.append(metric(y.iloc[test], fitted.predict(X.iloc[test])))
    print(f"{metric.__name__}: {100 * numpy.mean(scores):.1f}% (+/- {100 * numpy.std(scores):.1f}) over {len(folds)} folds")

    train, test = folds[0]
    _, gini, shapley = random_forest_rankings(X.iloc[train], y.iloc[train], X.iloc[test])
    rankings = {"RF": gini, "Shap": shapley, "SLDA": stepwise_lda(X.iloc[train], y.iloc[train])}

    rows = pandas.DataFrame([
        {"dimension": dimension, "method": method, "rank": rank + 1, "weight": args.top - rank}
        for method, ranking in rankings.items()
        for rank, dimension in enumerate(ranking[:args.top])
        ])
    rows.to_csv(f"{args.output}.csv", index=False)
    write_parallel_figure(rows, ["dimension", "method"], f"{args.output}.html")


if __name__ == "__main__":
    main()
