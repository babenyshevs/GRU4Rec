"""Sampling helpers and iterators for GRU4Rec training."""

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch


class SampleCache:
    """Pre-generates and caches negative samples for efficiency."""

    def __init__(
        self, n_sample, sample_cache_max_size, distr, device=torch.device("cuda:0")
    ):
        """Initialize the cache given a sampling distribution."""
        self.device = device
        self.n_sample = n_sample
        self.generate_length = (
            sample_cache_max_size // n_sample if n_sample > 0 else 0
        )
        self.distr = distr
        self._refresh()
        print(
            "Created sample store with {} batches of samples (type=GPU)".format(
                self.generate_length
            )
        )

    def _bin_search(self, arr, x):
        """Binary search over cumulative distribution ``arr``."""
        l = x.shape[0]
        a = torch.zeros(l, dtype=torch.int64, device=self.device)
        b = torch.zeros(l, dtype=torch.int64, device=self.device) + arr.shape[0]
        while torch.any(a != b):
            ab = torch.div((a + b), 2, rounding_mode="trunc")
            val = arr[ab]
            amask = val <= x
            a[amask] = ab[amask] + 1
            b[~amask] = ab[~amask]
        return a

    def _refresh(self):
        """Draw a new batch of negative samples."""
        if self.n_sample <= 0:
            return
        x = torch.rand(
            self.generate_length * self.n_sample,
            dtype=torch.float32,
            device=self.device,
        )
        self.neg_samples = self._bin_search(self.distr, x).reshape(
            (self.generate_length, self.n_sample)
        )
        self.sample_pointer = 0

    def get_sample(self):
        """Return a cached sample, refreshing the cache if needed."""
        if self.sample_pointer >= self.generate_length:
            self._refresh()
        sample = self.neg_samples[self.sample_pointer]
        self.sample_pointer += 1
        return sample


class SessionDataIterator:
    """Iterate through sessions yielding training batches."""

    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        n_sample: int = 0,
        sample_alpha: float = 0.75,
        sample_cache_max_size: int = 10000000,
        item_key: str = "ItemId",
        session_key: str = "SessionId",
        time_key: str = "Time",
        session_order: str = "time",
        device=torch.device("cuda:0"),
        itemidmap: Optional[pd.Series] = None,
    ):
        """Prepare iteration state and optional negative sampling cache."""
        self.device = device
        self.batch_size = batch_size
        if itemidmap is None:
            itemids = data[item_key].unique()
            self.n_items = len(itemids)
            self.itemidmap = pd.Series(
                data=np.arange(self.n_items, dtype="int32"),
                index=itemids,
                name="ItemIdx",
            )
        else:
            print("Using existing item ID map")
            self.itemidmap = itemidmap
            self.n_items = len(itemidmap)
            in_mask = data[item_key].isin(itemidmap.index.values)
            n_not_in = (~in_mask).sum()
            if n_not_in > 0:
                data = data.drop(data.index[~in_mask])
        self.sort_if_needed(data, [session_key, time_key])
        self.offset_sessions = self.compute_offset(data, session_key)
        if session_order == "time":
            self.session_idx_arr = np.argsort(
                data.groupby(session_key)[time_key].min().values
            )
        else:
            self.session_idx_arr = np.arange(len(self.offset_sessions) - 1)
        self.data_items = self.itemidmap[data[item_key].values].values
        if n_sample > 0:
            pop = data.groupby(item_key).size()
            pop = pop[self.itemidmap.index.values].values ** sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            distr = torch.tensor(pop, device=self.device, dtype=torch.float32)
            self.sample_cache = SampleCache(
                n_sample, sample_cache_max_size, distr, device=self.device
            )

    def sort_if_needed(self, data, columns, any_order_first_dim=False):
        """Ensure data are sorted by ``columns`` for efficient iteration."""
        is_sorted = True
        neq_masks = []
        for i, col in enumerate(columns):
            dcol = data[col]
            neq_masks.append(dcol.values[1:] != dcol.values[:-1])
            if i == 0:
                if any_order_first_dim:
                    is_sorted = is_sorted and (
                        dcol.nunique() == neq_masks[0].sum() + 1
                    )
                else:
                    is_sorted = is_sorted and np.all(
                        dcol.values[1:] >= dcol.values[:-1]
                    )
            else:
                is_sorted = is_sorted and np.all(
                    neq_masks[i - 1] | (dcol.values[1:] >= dcol.values[:-1])
                )
            if not is_sorted:
                break
        if is_sorted:
            print("The dataframe is already sorted by {}".format(", ".join(columns)))
        else:
            print(
                "The dataframe is not sorted by {}, sorting now".format(col)
            )
            t0 = time.time()
            data.sort_values(columns, inplace=True)
            t1 = time.time()
            print("Data is sorted in {:.2f}".format(t1 - t0))

    def compute_offset(self, data, column):
        """Compute index offsets for each session identifier."""
        offset = np.zeros(data[column].nunique() + 1, dtype=np.int32)
        offset[1:] = data.groupby(column).size().cumsum()
        return offset

    def __call__(self, enable_neg_samples, reset_hook=None):
        """Yield batches of input and target indices."""
        batch_size = self.batch_size
        iters = np.arange(batch_size)
        maxiter = iters.max()
        start = self.offset_sessions[self.session_idx_arr[iters]]
        end = self.offset_sessions[self.session_idx_arr[iters] + 1]
        finished = False
        valid_mask = np.ones(batch_size, dtype="bool")
        n_valid = self.batch_size
        while not finished:
            minlen = (end - start).min()
            out_idx = torch.tensor(
                self.data_items[start], requires_grad=False, device=self.device
            )
            for i in range(minlen - 1):
                in_idx = out_idx
                out_idx = torch.tensor(
                    self.data_items[start + i + 1],
                    requires_grad=False,
                    device=self.device,
                )
                if enable_neg_samples:
                    sample = self.sample_cache.get_sample()
                    y = torch.cat([out_idx, sample])
                else:
                    y = out_idx
                yield in_idx, y
            start = start + minlen - 1
            finished_mask = end - start <= 1
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
            maxiter += n_finished
            valid_mask = iters < len(self.offset_sessions) - 1
            n_valid = valid_mask.sum()
            if n_valid == 0:
                finished = True
                break
            mask = finished_mask & valid_mask
            sessions = self.session_idx_arr[iters[mask]]
            start[mask] = self.offset_sessions[sessions]
            end[mask] = self.offset_sessions[sessions + 1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if reset_hook is not None:
                finished = reset_hook(n_valid, finished_mask, valid_mask)
