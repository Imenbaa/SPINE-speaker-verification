"""Relate selected SPINE dimensions to acoustic descriptors (paper Section V).

Each dimension becomes a binary target by thresholding at 0.5. Classes are balanced by
random sampling, then four methods rank the descriptors that predict the target. Train
and test utterances never share a group, which should be the speaker. Prints each
surrogate's accuracy, and writes one row per dimension, method, and descriptor to
`<output>.csv` and the parallel categories figure to `<output>.html`.
"""
import argparse

import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

from common import load_embeddings, load_labels, write_parallel_figure
from selection import disr, random_forest_rankings, stepwise_lda


# Table III of the paper, as name prefixes of the eGeMAPS functionals.
FAMILIES = {
        "logRelF0": "F0", "F0": "F0",
        "F1": "F1", "F2": "F2", "F3": "F3",
        "mfcc": "MFCC",
        "jitter": "VoQ", "shimmer": "VoQ", "HNR": "VoQ",
        "Voiced": "Rhythm", "Mean": "Rhythm", "Stddev": "Rhythm", "Unvoiced": "Rhythm",
        "loudness": "Loudness", "equivalentSoundLevel": "Loudness",
        "alphaRatio": "Spectral", "slope": "Spectral", "spectralFlux": "Spectral", "hammarberg": "Spectral",
        }


def family_of(descriptor):
    return next(family for prefix, family in FAMILIES.items() if descriptor.startswith(prefix))


def balance(target, seed=0):
    """Return utterances with both classes randomly sampled down to the smaller class size."""
    size = target.value_counts().min()
    return pandas.concat([
        members.sample(size, random_state=seed) for _, members in target.groupby(target)
        ]).sort_index().index


def explain_dimension(descriptors, target, groups, top):
    """Return one dimension's top descriptors per method, and its surrogate model's test accuracy."""
    index = balance(target)
    X, y = descriptors.loc[index], target.loc[index]

    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups.loc[index]))
    forest, gini, shapley = random_forest_rankings(X.iloc[train], y.iloc[train], X.iloc[train])
    accuracy = accuracy_score(y.iloc[test], forest.predict(X.iloc[test]))

    return {"RF": gini, "Shap": shapley, "SLDA": stepwise_lda(X, y), "DISR": disr(X, y, k=top)}, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", help="pickle written by SPINE.EmbeddingWriter")
    parser.add_argument("descriptors", help="CSV written by extract_egemaps.py")
    parser.add_argument("labels", help="CSV with utterance_id and group columns")
    parser.add_argument("output", help="output path, without extension")
    parser.add_argument("--dimensions", type=int, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=3, help="descriptors kept per method")
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings)
    descriptors = pandas.read_csv(args.descriptors, index_col="utterance_id")
    groups = load_labels(args.labels)["group"]
    shared = embeddings.index.intersection(descriptors.index).intersection(groups.index)

    rows = []
    for dimension in args.dimensions:
        target = (embeddings.loc[shared, dimension] >= args.threshold).astype(int)
        if target.nunique() < 2:
            print(f"F{dimension}: skipped, only one class at threshold {args.threshold}")
            continue
        rankings, accuracy = explain_dimension(descriptors.loc[shared], target, groups, args.top)
        print(f"F{dimension}: surrogate test accuracy {100 * accuracy:.1f}%, {100 * (1 - target.mean()):.1f}% zeros")
        rows += [
            {"dimension": f"F{dimension}", "method": method, "rank": rank + 1, "descriptor": descriptor,
             "family": family_of(descriptor), "weight": args.top - rank}
            for method, ranking in rankings.items()
            for rank, descriptor in enumerate(ranking[:args.top])
            ]

    rows = pandas.DataFrame(rows)
    rows.to_csv(f"{args.output}.csv", index=False)
    write_parallel_figure(rows, ["dimension", "descriptor", "family"], f"{args.output}.html")


if __name__ == "__main__":
    main()
