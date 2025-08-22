"""Utilities for preparing GRU4Rec datasets.

This module provides helper functions for working with session-based
recommendation datasets. Data can be loaded directly from a Databricks table
into a :class:`pandas.DataFrame` using :func:`load_databricks_table`, then split
into train, validation and test sets with :func:`train_valid_test_split`.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_COL_NAMES = {
    "session_key": "SessionId",
    "item_key": "ItemId",
    "time_key": "Time",
}


def load_databricks_table(
    table_name: str,
    session_key: str = DEFAULT_COL_NAMES["session_key"],
    item_key: str = DEFAULT_COL_NAMES["item_key"],
    time_key: str = DEFAULT_COL_NAMES["time_key"],
) -> pd.DataFrame:
    """Load a Databricks table into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    table_name : str
        Name of the table accessible to Spark.
    session_key, item_key, time_key : str, optional
        Column names to select from the table. Defaults correspond to the
        expected column names for GRU4Rec.

    Returns
    -------
    pandas.DataFrame
        Data frame with the specified columns converted to appropriate types.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "pyspark is required to load data from a Databricks table"
        ) from exc

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "No active Spark session found. This function must be called "
            "within a Databricks environment with Spark available."
        )

    sdf = spark.table(table_name).select(session_key, item_key, time_key)
    df = sdf.toPandas()
    df[session_key] = df[session_key].astype("float32")
    df[item_key] = df[item_key].astype("float32")
    df[time_key] = pd.to_datetime(df[time_key])
    return df


def _raise_missing(column: str, kind: str, arg_name: str) -> None:
    """Raise a helpful error for missing ``column`` in the data frame."""
    default_name = DEFAULT_COL_NAMES[arg_name]
    raise ValueError(
        f'ERROR. The column specified for {kind} "{column}" is not in the data frame. '
        f'The default column name is "{default_name}", but you can specify otherwise by setting '
        f'the `{arg_name}` parameter of the model.'
    )


def _validate_columns(
    df: pd.DataFrame, session_key: str, item_key: str, time_key: str
) -> None:
    """Verify that the required columns exist in ``df``."""
    for column, kind, arg in [
        (session_key, "session IDs", "session_key"),
        (item_key, "item IDs", "item_key"),
        (time_key, "time", "time_key"),
    ]:
        if column not in df.columns:
            _raise_missing(column, kind, arg)


def train_valid_test_split(
    data: pd.DataFrame,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    session_key: str = "SessionId",
    item_key: str = "ItemId",
    time_key: str = "Time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``data`` into train, validation and test sets.

    The split is performed on a session level. Sessions are ordered by their
    last interaction timestamp and divided into train/validation/test by the
    provided fractions.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing at least ``session_key``, ``item_key`` and
        ``time_key`` columns.
    valid_fraction : float, optional
        Fraction of sessions to include in the validation set.
    test_fraction : float, optional
        Fraction of sessions to include in the test set.
    session_key, item_key, time_key : str
        Column names in ``data``.

    Returns
    -------
    (train, valid, test) : tuple of pandas.DataFrame
        Split datasets.
    """

    _validate_columns(data, session_key, item_key, time_key)

    # Determine session order based on the last event timestamp.
    session_end = data.groupby(session_key)[time_key].max().sort_values()
    sessions_ordered = session_end.index.tolist()

    n_sessions = len(sessions_ordered)
    n_test = int(n_sessions * test_fraction)
    n_valid = int(n_sessions * valid_fraction)

    train_cutoff = n_sessions - n_test - n_valid
    valid_cutoff = n_sessions - n_test

    train_sessions = sessions_ordered[:train_cutoff]
    valid_sessions = sessions_ordered[train_cutoff:valid_cutoff]
    test_sessions = sessions_ordered[valid_cutoff:]

    train = data[data[session_key].isin(train_sessions)]
    valid = data[data[session_key].isin(valid_sessions)]
    test = data[data[session_key].isin(test_sessions)]

    return train, valid, test

