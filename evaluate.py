"""Compute sparsity and speaker verification EER from a predicted embeddings pickle.

Sparsity is the percentage of zero values over D x N. EER uses cosine scoring on a
trial list with lines `<1|0> <utterance_id> <utterance_id>`, the format of VoxCeleb1-O.
"""
import argparse
import pickle

import numpy


def sparsity(embeddings):
    return 100 * float(numpy.mean(embeddings == 0))


def equal_error_rate(scores, targets):
    """Return the EER in percent, at the threshold where miss and false alarm rates cross."""
    order = numpy.argsort(scores)
    targets = numpy.asarray(targets, dtype=bool)[order]
    miss = numpy.cumsum(targets) / targets.sum()
    false_alarm = 1 - numpy.cumsum(~targets) / (~targets).sum()
    index = numpy.argmin(numpy.abs(miss - false_alarm))
    return 100 * float((miss[index] + false_alarm[index]) / 2)


def score_trials(embeddings, labels, trials_path):
    """Return the cosine score and target flag of each trial."""
    norms = numpy.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / numpy.where(norms == 0, 1, norms)
    index = {label: i for i, label in enumerate(labels)}

    scores, targets = [], []
    with open(trials_path) as f:
        for line in f:
            target, first, second = line.split()
            scores.append(float(normalized[index[first]] @ normalized[index[second]]))
            targets.append(target == "1")
    return numpy.array(scores), numpy.array(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", help="pickle written by SPINE.EmbeddingWriter")
    parser.add_argument("trials", help="trial list")
    args = parser.parse_args()

    with open(args.embeddings, "rb") as f:
        data = pickle.load(f)

    scores, targets = score_trials(data["embedding"], data["label"], args.trials)
    print(f"Sparsity: {sparsity(data['embedding']):.1f}%")
    print(f"EER: {equal_error_rate(scores, targets):.2f}%")


if __name__ == "__main__":
    main()
