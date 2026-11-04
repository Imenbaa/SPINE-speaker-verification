# Interpretable Sparse and Binary Representations for Speaker Verification

Code for the method in the paper by I. Ben-Amor, M. Tahon, J.-F. Bonastre and N. Dugué.

## Method

| Stage | What it does | Code |
|---|---|---|
| 1. Sparse and binary embeddings | Trains a SPINE autoencoder on pretrained x-vectors, with reconstruction, average sparsity, and partial sparsity losses. Activations are thresholded at 0.5 after training. Sparsity and cosine EER measure the result. | `SPINE.py`, `evaluate.py` |
| 2. Probing | Classifies gender and emotion from the embeddings, then ranks dimensions with RF Gini, SHAP, and stepwise LDA to select the ones linked to each task. | `Explainability/probe.py` |
| 3. Explanation | Turns each selected dimension into a binary target and predicts it from eGeMAPS descriptors. RF Gini, SHAP, stepwise LDA, and DISR then show which descriptor families explain it. | `Explainability/explain.py` |

## Installation

Tested with Python 3.10:

```sh
pip install -r requirements.txt
```

## X-vectors

SPINE takes pretrained 256-dimensional x-vectors as input. They come from the ResNet34 in [stklia](https://github.com/Chaanks/stklia), whose [VoxCeleb example](https://github.com/Chaanks/stklia/tree/master/exemples/voxceleb) includes a checkpoint.

Prepare your Kaldi feature folders with stklia's [VoxCeleb recipe](https://github.com/Chaanks/stklia/tree/master/recipes/Voxceleb), then run this from the stklia folder for both VoxCeleb-2 dev and VoxCeleb-1. Its default output format needs Kaldi's `copy-vector` on your `PATH`:

```sh
python extract.py --modeldir exemples/voxceleb/model_dir --checkpoint 5800 --data path/to/kaldi/data/folder
```

Each run writes `exemples/voxceleb/model_dir/xvectors/<folder_name>/xvectors.scp`. Set these paths in `conf/data.yaml`:

| Key | X-vectors |
|---|---|
| `train_scp` | VoxCeleb-2 |
| `val_scp` | VoxCeleb-1 |
| `predict_scp` | VoxCeleb-1 |

## Training

SLURM settings are in `submit.sh`. To train the default configuration:

```sh
sbatch submit.sh --config conf/SPINE.yaml --config conf/data.yaml --config conf/AdamW.yaml --trainer.logger.init_args.version spine_500_1_1_1
```

Logs and checkpoints are written to `lightning_logs/<version>/`.

### Paper configurations

All use `D = 500`. `conf/SPINE.yaml` holds the first one; add the flags to select another:

| λr | λas | λps | Flags |
|---|---|---|---|
| 1 | 1 | 1 | none |
| 1 | 50 | 10 | `--model.init_args.as_lambda 50 --model.init_args.ps_lambda 10` |
| 1 | 1 | 10 | `--model.init_args.ps_lambda 10` |

## Evaluation

Extract embeddings from a trained checkpoint. The default output is continuous:

```sh
python CLI.py predict --config conf/trainer.yaml --config conf/SPINE.yaml --config conf/data.yaml --config conf/predict.yaml --ckpt_path lightning_logs/spine_500_1_1_1/checkpoints/last.ckpt
```

Add `--model.init_args.binarize true` for binary embeddings, which threshold activations at 0.5 after training. Change the output file with `--trainer.callbacks.init_args.output_path`.

Then compute sparsity and EER with cosine scoring. The trial list is stklia's `veri_pairs` file, with lines of the form `<1|0> <utterance_id> <utterance_id>`:

```sh
python evaluate.py embeddings.pickle path/to/veri_pairs
```

## Probing and explainability

These scripts use a separate environment, tested with Python 3.10. Run them from `Explainability/`, with continuous embeddings from the evaluation step:

```sh
pip install -r requirements.txt
```

Probing and explanation take a labels CSV keyed by the same utterance IDs as the embeddings. Folds and splits never share a `group`:

| Task | `group` | `label` |
|---|---|---|
| Gender | speaker | gender |
| Emotion | session | emotion |

### Probe a task and select dimensions

```sh
python probe.py embeddings.pickle gender_labels.csv gender gender_dimensions
python probe.py embeddings.pickle emotion_labels.csv emotion emotion_dimensions
```

Add `--binarize` to probe binary embeddings. Each command prints the cross-validated score: accuracy with a random forest for gender, and unweighted average recall with an SVM for emotion. It writes dimension rankings from RF Gini, SHAP, and stepwise LDA to `<output>.csv`, and a figure to `<output>.html`.

### Explain the selected dimensions

Extract the 88 eGeMAPS descriptors from a `<utterance_id> <wav_path>` list, then pass the dimensions you selected from `gender_dimensions.csv`. The numbers below are placeholders:

```sh
python extract_egemaps.py wav.scp descriptors.csv
python explain.py embeddings.pickle descriptors.csv gender_labels.csv gender_explanation --dimensions 12 34 56
```

Each dimension becomes a binary target with a threshold at 0.5. Descriptors are ranked by RF Gini, SHAP, stepwise LDA, and DISR, then grouped into families. The command prints each surrogate model's test accuracy; a ranking is only meaningful when that accuracy is well above 50%.

## License

This project is released under the [MIT License](LICENSE).
