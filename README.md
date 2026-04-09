# Personality-Aware Sequential Self-Attention Learning for Next Item Recommendations

This repository contains the implementations of the models used in the coursework "Personality-Aware Sequential Self-Attention Learning for Next Item Recommendations" as well as the [report](coursework.pdf). All methodology and evaluations details can be found there.

## Installation

1. Install all required packages:

```bash
pip install -r requirements.txt
```

2. Install `pre-commit`:

```bash
pre-commit install
```

## Training

To train and reproduce the results for the models run the following command:

```bash
python train.py -cn=CONFIG_NAME
```

where `CONFIG_NAME` user the format `{TYPE}_{DATASET}`, where `{TYPE}` is `sasrec` or `user_sasrec` and `{DATASET}` is one of `yambda`, `gowalla`, `yelp` or `cosmetics`. There are also extra configs which are used in the paper (e.g., `user_sasrec_gowalla_tucker` config).

Validation splits configs are provided as `src/configs/datasets/{DATASET}.yaml` and test splits as `src/configs/datasets/{DATASET}_test.yaml`.

Dataset download links are provided in the respective classes in `src/datasets`.

## Credits

This repository uses the following [project template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
