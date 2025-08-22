"""Utilities for loading GRU4Rec datasets.

This module provides helper functions for reading training or evaluation
files stored either as tab separated values (TSV) or as pickled
``pandas.DataFrame`` objects.  Each reader validates that the required
columns are present and returns the data as a ``DataFrame`` so that other
scripts can reuse the same loading logic without duplication.

Example
-------
>>> from gru4rec.data_loader import load_data
>>> df = load_data("train.tsv")
"""
from __future__ import annotations

import pandas as pd
import joblib

DEFAULT_COL_NAMES = {
    "session_key": "SessionId",
    "item_key": "ItemId",
    "time_key": "Time",
}


def _raise_missing(column: str, kind: str, fname: str, arg_name: str) -> None:
    default_name = DEFAULT_COL_NAMES[arg_name]
    raise ValueError(
        f'ERROR. The column specified for {kind} "{column}" is not in the data file ({fname}). '
        f'The default column name is "{default_name}", but you can specify otherwise by setting '
        f'the `{arg_name}` parameter of the model.'
    )


def _validate_columns(
    df: pd.DataFrame, fname: str, session_key: str, item_key: str, time_key: str
) -> None:
    for column, kind, arg in [
        (session_key, "session IDs", "session_key"),
        (item_key, "item IDs", "item_key"),
        (time_key, "time", "time_key"),
    ]:
        if column not in df.columns:
            _raise_missing(column, kind, fname, arg)


def _validate_header(
    header: list[str], fname: str, session_key: str, item_key: str, time_key: str
) -> None:
    for column, kind, arg in [
        (session_key, "session IDs", "session_key"),
        (item_key, "item IDs", "item_key"),
        (time_key, "time", "time_key"),
    ]:
        if column not in header:
            _raise_missing(column, kind, fname, arg)


def read_pickle(
    fname: str,
    session_key: str = "SessionId",
    item_key: str = "ItemId",
    time_key: str = "Time",
) -> pd.DataFrame:
    """Load a pickled ``DataFrame`` and validate required columns."""
    print(f"Loading data from pickle file: {fname}")
    data = joblib.load(fname)
    _validate_columns(data, fname, session_key, item_key, time_key)
    return data


def read_tsv(
    fname: str,
    session_key: str = "SessionId",
    item_key: str = "ItemId",
    time_key: str = "Time",
) -> pd.DataFrame:
    """Load a TAB separated file into a ``DataFrame`` and validate columns."""
    with open(fname, "rt") as f:
        header = f.readline().strip().split("\t")
    _validate_header(header, fname, session_key, item_key, time_key)
    print(f"Loading data from TAB separated file: {fname}")
    data = pd.read_csv(
        fname,
        sep="\t",
        usecols=[session_key, item_key, time_key],
        dtype={session_key: "int32", item_key: "str"},
    )
    _validate_columns(data, fname, session_key, item_key, time_key)
    return data


def load_data(
    fname: str,
    session_key: str = "SessionId",
    item_key: str = "ItemId",
    time_key: str = "Time",
) -> pd.DataFrame:
    """Load a dataset from ``fname`` and ensure required columns are present.

    Parameters
    ----------
    fname : str
        Path to the dataset.  Files ending with ``.pickle`` are expected to be
        pickled :class:`pandas.DataFrame` objects.  Other extensions are
        interpreted as TAB separated files.
    session_key, item_key, time_key : str
        Column names for sessions, items and timestamps.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset containing at least the specified columns.

    Raises
    ------
    ValueError
        If any of the required columns are missing from ``fname``.
    """
    if fname.endswith(".pickle"):
        return read_pickle(fname, session_key, item_key, time_key)
    return read_tsv(fname, session_key, item_key, time_key)
