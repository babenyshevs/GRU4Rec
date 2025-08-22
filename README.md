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

