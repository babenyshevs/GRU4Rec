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

Use :func:`gru4rec.train_valid_test_split` to split the DataFrame into
training, validation and test sets.  The validation split is intended for
hyperparameter optimisation.

```python
from gru4rec import train_valid_test_split
train, valid, test = train_valid_test_split(df)
```

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

