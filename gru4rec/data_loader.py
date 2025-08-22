"""Utilities for preparing GRU4Rec datasets.

This module assumes that data is already loaded into a ``pandas.DataFrame``
within an environment such as Databricks.  The :func:`train_valid_test_split`
function performs a temporal split of the data into training, validation and
test sets based on session end times.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_COL_NAMES = {
    "session_key": "SessionId",
    "item_key": "ItemId",
    "time_key": "Time",
}


def _raise_missing(column: str, kind: str, arg_name: str) -> None:
    default_name = DEFAULT_COL_NAMES[arg_name]
    raise ValueError(
        f'ERROR. The column specified for {kind} "{column}" is not in the data frame. '
        f'The default column name is "{default_name}", but you can specify otherwise by setting '
        f'the `{arg_name}` parameter of the model.'
    )


def _validate_columns(
    df: pd.DataFrame, session_key: str, item_key: str, time_key: str
) -> None:
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

