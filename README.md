# GRU4Rec

Official PyTorch implementation of **GRU4Rec**, a session-based recommender
system built on gated recurrent units. The network predicts the next item in a
session and supports both pointwise and pairwise ranking objectives.

## Features

- Pure PyTorch code validated against the original implementation.
- Configurable stack of `GRUCell` layers with optional dropout.
- Item representations can share weights with the output layer, use a separate
  embedding matrix or learn input weights directly.
- Losses: `cross-entropy` with softmax or `bpr-max` with optional ELU.
- Negative sampling, popularity based weighting and GPU support.

## Repository Layout

| Path | Description |
| --- | --- |
| `gru4rec/` | Core library: model, trainer, optimizers and data utilities. |
| `run.py` | Train/evaluate models from the command line. |
| `paropt.py` | Hyperparameter optimisation with Optuna. |
| `config/` | Example JSON configs for the scripts. |
| `paramfiles/` | Parameter presets for common datasets. |
| `paramspaces/` | Search spaces for hyperparameter tuning. |

## Data Format

Data sets are tab separated files (or pickled `pandas.DataFrame` objects)
containing the following columns:

| Column | Description |
| --- | --- |
| `SessionId` | Unique session identifier. |
| `ItemId` | Integer item identifier. |
| `Time` | Timestamp in seconds. |

Custom column names can be supplied through the `session_key`, `item_key` and
`time_key` options.

## Configuration

`run.py` and `paropt.py` read their settings from JSON (or YAML) files. By
default they use:

- `config/run.json`
- `config/paropt.json`

Override the locations with the `GRU4REC_RUN_CONFIG` and
`GRU4REC_PAROPT_CONFIG` environment variables.

Example `config/run.json`:

```json
{
  "path": "path/to/train.tsv",
  "parameter_file": "paramfiles/rsc15_xe_shared_100_best.py",
  "test": ["path/to/test.tsv"],
  "device": "cuda:0"
}
```

### Parameter specification

Exactly one of the following must be provided when training or evaluating:

1. **Parameter string** – comma separated `name=value` pairs

   ```bash
   python run.py --parameter_string "loss=cross-entropy,layers=100,n_epochs=10" \
                 path/to/train.tsv --test path/to/test.tsv
   ```

2. **Parameter file** – Python module defining `gru4rec_params`

   ```python
   from collections import OrderedDict
   gru4rec_params = OrderedDict([
       ("loss", "bpr-max"),
       ("layers", [480]),
       ("learning_rate", 0.07),
       ("n_epochs", 10)
   ])
   ```

3. **Serialized model** – `--load_model` to evaluate or continue training

## Basic Usage

1. Prepare data as described above.
2. Train and evaluate a model:

   ```bash
   python run.py path/to/train.tsv --test path/to/test.tsv \
        --parameter_file paramfiles/yoochoose_xe_shared_best.py
   ```

3. Save a trained model:

   ```bash
   python run.py path/to/train.tsv --parameter_string "loss=cross-entropy,layers=100" \
        --save_model model.pt
   ```

4. Evaluate an existing model:

   ```bash
   python run.py model.pt --load_model --test path/to/test.tsv
   ```

## Hyperparameter Tuning

`paropt.py` uses Optuna to explore the search space defined in
`config/paropt.json`:

```bash
python paropt.py path/to/train.tsv path/to/valid.tsv \
     -opf config/paropt.json
```

Use `fixed_parameters` in the config to lock certain values and
`optuna_parameter_file` to describe the search space.

## References

- Balázs Hidasi et al., "Session-based Recommendations with Recurrent Neural
  Networks", ICLR 2016.
- Balázs Hidasi & Alexandros Karatzoglou, "Recurrent Neural Networks with
  Top-k Gains for Session-based Recommendations", CIKM 2018.

**License:** Free for research and education. Contact the authors for
commercial use.

