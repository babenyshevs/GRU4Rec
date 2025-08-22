# GRU4Rec

Official PyTorch implementation of **GRU4Rec**, a session-based recommender
system built on gated recurrent units. The network predicts the next item in a
session and supports both pointwise and pairwise ranking objectives.

## Data preparation

Data must be available as a pandas ``DataFrame`` (for example in Databricks)
with the following columns:

| Column | Description |
| --- | --- |
| ``SessionId`` | Unique session identifier. |
| ``ItemId`` | Integer item identifier. |
| ``Time`` | Timestamp in seconds. |

If your interactions are stored in a Databricks table, you can load them
directly into a DataFrame:

```python
from gru4rec import load_databricks_table
df = load_databricks_table("schema.table")
```

Use :func:`gru4rec.train_valid_test_split` to split the DataFrame into
training, validation and test sets.  The validation split is intended for
hyperparameter optimisation.

```python
from gru4rec import train_valid_test_split
train, valid, test = train_valid_test_split(df)
```

## Configuration

Model, data-loading, training and evaluation parameters can be stored in a YAML
configuration file. An extended example is provided in
`config/example.yaml`:

```yaml
model:
  layers: [10]
  loss: cross-entropy
  batch_size: 2
  dropout_p_embed: 0.0
  dropout_p_hidden: 0.0
  learning_rate: 0.05
  momentum: 0.0
  sample_alpha: 0.5
  n_sample: 2048
  embedding: 0
  constrained_embedding: true
  n_epochs: 1
  bpreg: 1.0
  elu_param: 0.5
  logq: 0.0
  device: cpu

data:
  session_key: SessionId
  item_key: ItemId
  time_key: Time

data_split:
  valid_fraction: 0.1
  test_fraction: 0.1

training:
  sample_cache_max_size: 10000000
  compatibility_mode: true

evaluation:
  cutoff: [20]
  batch_size: 512
  mode: conservative

paths:
  model_save: model.pth
  model_load: model.pth
```

### Configuration options

**Model**

- `layers`: list defining the size of each GRU layer. Values between 50 and 512
  units per layer are common.
- `loss`: training objective, either `cross-entropy` (pointwise) or `bpr-max`
  (pairwise).
- `batch_size`: number of sessions processed in parallel; typically 32–512.
- `dropout_p_embed`: dropout applied to item embeddings; 0.0–0.5.
- `dropout_p_hidden`: dropout on hidden states; 0.0–0.5.
- `learning_rate`: optimizer step size. Useful range is 1e-4 to 1e-1.
- `momentum`: momentum coefficient for the optimizer; 0.0–0.99.
- `sample_alpha`: exponent for popularity-based negative sampling. `0` gives
  uniform sampling, while values in [0,1] bias towards popular items.
- `n_sample`: number of negative samples per positive example. Set to `0` to
  disable sampling; values up to a few thousand are typical.
- `embedding`: dimension of a separate item embedding matrix. `0` uses GRU
  weights instead.
- `constrained_embedding`: tie input and output embeddings when `true`.
- `n_epochs`: number of training epochs.
- `bpreg`: regularization strength for the BPR-max loss; usually 0–2.
- `elu_param`: ELU parameter for the BPR-max loss; 0.0–1.0.
- `logq`: logarithmic popularity correction for cross-entropy loss; 0 disables
  correction.
- `device`: computation device such as `cpu` or `cuda:0`.

**Data**

- `session_key`, `item_key`, `time_key`: column names in the input data frame
  identifying sessions, items and timestamps.

**Data split**

- `valid_fraction`, `test_fraction`: fractions of sessions reserved for
  validation and testing respectively; values between 0 and 1.

**Training**

- `sample_cache_max_size`: maximum number of cached negative samples. Adjust to
  fit available memory; ranges from 1e6 to 1e8 are typical.
- `compatibility_mode`: if `true`, reset weights to match the original
  GRU4Rec initialization.

**Evaluation**

- `cutoff`: list of ranking cutoffs (e.g. `[20]`).
- `batch_size`: number of sessions per evaluation batch; often 128–1024.
- `mode`: tie-handling strategy for ranking; `conservative`, `standard` or
  `median`.

**Paths**

- `model_save`: file path where the trained model is saved.
- `model_load`: path of a model to load before training or evaluation.

Use the helper functions to load the configuration, split the data and build
the model:

```python
from gru4rec import (
    load_config,
    build_model,
    split_data,
    evaluate,
)

config = load_config("config/example.yaml")
train, valid, test = split_data(df, config)
gru = build_model(config)
gru.fit(train, **config["training"], **config["data"])
recall, mrr = evaluate(gru, test, config)
```

The notebook `example.ipynb` demonstrates this workflow end-to-end.

## Training

```python
from gru4rec import GRU4Rec

gru = GRU4Rec(layers=[100])
gru.fit(train)
```

Evaluate the model using ``evaluation.batch_eval`` on the ``test`` DataFrame.

## References

- Balázs Hidasi et al., "Session-based Recommendations with Recurrent Neural
  Networks", ICLR 2016.
- Balázs Hidasi & Alexandros Karatzoglou, "Recurrent Neural Networks with
  Top-k Gains for Session-based Recommendations", CIKM 2018.

**License:** Free for research and education. Contact the authors for
commercial use.

