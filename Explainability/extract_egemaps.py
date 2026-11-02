"""Extract the 88 eGeMAPS functionals for each utterance of a `<utterance_id> <wav_path>` list."""
import argparse

import opensmile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_scp", help="file with lines `<utterance_id> <wav_path>`")
    parser.add_argument("output_csv")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.wav_scp) as f:
        utterance_ids, paths = zip(*(line.split(maxsplit=1) for line in f if line.strip()))

    smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=args.workers,
            )
    descriptors = smile.process_files([path.strip() for path in paths])
    descriptors.index = list(utterance_ids)
    descriptors.to_csv(args.output_csv, index_label="utterance_id")


if __name__ == "__main__":
    main()
