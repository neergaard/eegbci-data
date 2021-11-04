import argparse
import logging
import os
import random
from itertools import compress
from typing import Callable, Optional, Tuple

import numpy as np
from h5py import File
from joblib import Memory, delayed
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset

from .utils import ParallelExecutor


logger = logging.getLogger("mne")
memory = Memory("data/.cache", mmap_mode="r", verbose=0)
SCALERS = {"robust": preprocessing.RobustScaler, "standard": preprocessing.StandardScaler}


@memory.cache
def _initialize_record(filename: str = None, scaling: str = None, sequence_length: int = None) -> dict:
    if scaling in SCALERS.keys():
        scaler = SCALERS[scaling]()
    else:
        scaler = None

    with File(filename, "r") as h5:
        X = h5["signals"] or h5["waveforms"]
        y = h5["targets/labels"] or h5["tasks/labels"]
        t = h5["targets/onehot"] or h5["tasks/onehot"]
        metadata = dict(h5.attrs)
        record_id = metadata["id"]

        N, C, T = X.shape
        K = t.shape[-1]

        # Perform scaling
        if scaler:
            scaler.fit(X[:].transpose(1, 0, 2).reshape((C, N * T)).T)

        # Get class counts and locations of tasks
        class_counts = np.bincount(y[:].flatten() - 1, minlength=K)  # y labels start from 1
        class_indices = {v: np.where(y[:] == v)[0] for v in np.unique(y)}

        # Get number of sequences in file
        if isinstance(sequence_length, str) and sequence_length == "full":
            sequence_counts = 1
        else:
            sequence_counts = N - sequence_length + 1

        index_to_record = [{"record": record_id, "idx": x} for x in range(sequence_counts)]

    return dict(
        metadata=(record_id, metadata),
        scaler=(record_id, scaler),
        class_indices=(record_id, class_indices),
        class_counts=(record_id, class_counts),
        sequence_counts=(record_id, sequence_counts),
        index_to_record=index_to_record,
        data_shape=(N, C, T),
    )


class EEGBCIDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        balanced_sampling: Optional[bool] = None,
        cv: Optional[int] = None,
        cv_idx: Optional[int] = None,
        eval_ratio: Optional[float] = None,
        max_eval_records: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_jobs: Optional[int] = None,
        n_records: Optional[int] = None,
        scaling: Optional[str] = None,
        sequence_length: Optional[int] = None,
        transforms: Optional[list[Callable]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.balanced_sampling = balanced_sampling
        self.cv = cv
        self.cv_idx = cv_idx
        self.eval_ratio = eval_ratio
        self.max_eval_records = max_eval_records
        self.n_channels = n_channels
        self.n_jobs = n_jobs
        self.n_records = n_records
        self.scaling = scaling
        self.sequence_length = (
            sequence_length if isinstance(sequence_length, str) and sequence_length == "full" else int(sequence_length)
        )
        self.transforms = transforms
        self.kf = KFold(n_splits=self.cv) if self.cv > 1 else None
        self.records = sorted(os.listdir(self.data_dir))[: self.n_records]
        self.n_records = self.n_records or len(self.records)

        # Get information about data
        logger.info(f"Prefetching study metadata using {self.n_jobs} workers:")
        sorted_data = ParallelExecutor(n_jobs=self.n_jobs, prefer="threads")(total=len(self.records))(
            delayed(_initialize_record)(
                filename=os.path.join(self.data_dir, record),
                scaling=self.scaling,
                sequence_length=self.sequence_length,
            )
            for record in self.records
        )
        logger.info("Prefetching finished.")
        self.class_counts = sum([s["class_counts"][1] for s in sorted_data])
        self.class_indices = dict([s["class_indices"] for s in sorted_data])
        self.index_to_record = [sub for s in sorted_data for sub in s["index_to_record"]]
        self.metadata = dict([s["metadata"] for s in sorted_data])
        self.scalers = dict([s["scaler"] for s in sorted_data])
        self.sequence_counts = dict([s["sequence_counts"] for s in sorted_data])
        self.data_shape = sorted_data[0]["data_shape"][1:]

    def __len__(self) -> int:
        return sum(self.sequence_counts.values())

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, int]:

        if self.balanced_sampling:
            # Sample a subject
            current_subject = np.random.choice(self.records).split(".")[0]

            # Sample a task
            current_task = np.random.choice([k for k, v in self.class_indices[current_subject].items() if v.any()])

            # Sample a task location
            current_task_loc = np.random.choice(self.class_indices[current_subject][current_task])

            # Sample a sequence around the task location
            N = self.metadata[current_subject]["n_epochs"]
            current_sequence_start = np.random.choice(
                np.arange(
                    np.max([0, current_task_loc - self.sequence_length + 1]),
                    np.min([N - self.sequence_length + 1, current_task_loc + 1]),
                )
            )
            current_sequence = slice(current_sequence_start, current_sequence_start + self.sequence_length)
        else:
            current_subject = self.index_to_record[idx]["record"]
            if isinstance(self.sequence_length, str) and self.sequence_length == "full":
                current_sequence = slice(None)
            else:
                current_sequence = slice(
                    self.index_to_record[idx]["idx"], self.index_to_record[idx]["idx"] + self.sequence_length
                )

        # Grab data
        with File(os.path.join(self.data_dir, current_subject + ".h5"), "r") as h5:
            X = h5["signals"] or h5["waveforms"]
            y = h5["targets/labels"] or h5["tasks/labels"]
            t = h5["targets/onehot"] or h5["tasks/onehot"]
            x = X[current_sequence].astype("float32")
            t = t[current_sequence].astype("uint8")

        # Reshape data from (N, C, T) -> (C, N x T)
        N, C, T = x.shape
        x = x.transpose(1, 0, 2).reshape(C, N * T)
        t = t.T

        # Possibly perform scaling
        scaler = self.scalers[current_subject]
        if scaler:
            x = scaler.transform(x.T).T

        # Possibly do transforms
        if self.transforms is not None:
            for transform in self.transforms:
                x = transform(x)

        return x, t, current_subject, current_sequence

    def _shuffle_records(self) -> None:
        random.shuffle(self.records)

    def _split_data(self) -> Tuple[Dataset, Dataset]:
        self._shuffle_records()
        if self.kf:
            self.train_idx, self.eval_idx = list(self.kf.split(range(self.n_records)))[self.cv_idx]
        else:
            n_eval = min(int(self.n_records * self.eval_ratio), self.max_eval_records)
            self.train_idx = np.arange(n_eval, self.n_records)
            self.eval_idx = np.arange(0, n_eval)
        self.train_data = EEGBCISubset(self, self.train_idx, balanced_sampling=self.balanced_sampling, name="train")
        self.eval_data = EEGBCISubset(self, self.eval_idx, name="eval")

        return self.train_data, self.eval_data

    def __str__(self) -> str:
        return f"""

EEGBCI Dataset
==============

Parameters:
------------------------------------------------------------------------------
\tBalanced sampling:        {self.balanced_sampling}
\tCross-validation folds:   {self.cv}
\tCurrent CV fold:          {self.cv_idx}
\tData directory:           {self.data_dir}
\tEval ratio:               {self.eval_ratio}
\tNumber of channels:       {self.data_shape[0]}
\tNumber of records:        {self.n_records}
\tNumber of sequences:      {len(self)}
\tScaling type:             {self.scaling}
\tSequence length:          {self.sequence_length * 5} min
\tTransforms:               {self.transforms}
==============================================================================
"""

    def __repr__(self) -> str:
        return f"EEGBCIDataset({self.data_dir}, {self.balanced_sampling}, {self.cv}, {self.cv_idx}, {self.eval_ratio}, {self.max_eval_records}, {self.n_channels}, {self.n_jobs}, {self.n_records}, {self.scaling}, {self.sequence_length}, {self.transforms})"

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        # fmt: off
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        dataset_group = parser.add_argument_group("dataset")
        dataset_group.add_argument("-d", "--data_dir", default="./data/processed", type=str, help="Location of H5 dataset files.")
        dataset_group.add_argument("--balanced_sampling", default=None, action="store_true", help="Whether to balance batches.")
        dataset_group.add_argument("--cv", default=1, type=int, help="Number of CV folds.")
        dataset_group.add_argument("--cv_idx", default=0, type=int, help="Specific CV fold index.")
        dataset_group.add_argument("--eval_ratio", default=0.1, type=float, help="Ratio of validation subjects.")
        dataset_group.add_argument("--max_eval_records", default=100, type=int, help="Max. number of subjects to use for eval.")
        dataset_group.add_argument("--n_channels", default=None, type=int, help="Number of applied EEG channels.")
        dataset_group.add_argument("--n_jobs", default=-1, type=int, help="Number of parallel jobs to run for prefetching.")
        dataset_group.add_argument("--n_records", default=None, type=int, help="Total number of records to use.")
        dataset_group.add_argument("--scaling", default="robust", choices=["robust", "standard"], help="How to scale EEG data.")
        dataset_group.add_argument("--sequence_length", default=10, help="Number of sequences in each batch element.")
        # fmt: on
        return parser


class EEGBCISubset(Dataset):
    def __init__(self, dataset, record_indices, balanced_sampling=False, name=None):
        self.dataset = dataset
        self.record_indices = record_indices
        self.balanced_sampling = balanced_sampling
        self.name = name
        self.subset_records = [self.dataset.records[idx].split(".")[0] for idx in self.record_indices]
        self.sequence_indices = self.__get_subset_indices()

    def __get_subset_indices(self):
        records = set(self.subset_records)
        t = list(map(lambda x: x["record"] in records, self.dataset.index_to_record))
        return list(compress(range(len(t)), t))

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        return self.dataset[self.sequence_indices[idx]]

    def __str__(self):
        return f"""

EEGBCI Subset
==============

Parameters:
------------------------------------------------------------------------------
\tBalanced sampling:        {self.dataset.balanced_sampling}
\tCross-validation folds:   {self.dataset.cv}
\tCurrent CV fold:          {self.dataset.cv_idx}
\tData directory:           {self.dataset.data_dir}
\tEval ratio:               {self.dataset.eval_ratio}
\tNumber of channels:       {self.dataset.data_shape[0]}
\tNumber of records:        {len(self.subset_records)}
\tNumber of sequences:      {len(self)}
\tScaling type:             {self.dataset.scaling}
\tSequence length:          {self.dataset.sequence_length * 5} min
\tSubset:                   {self.name}
==============================================================================
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", default="./data/processed", type=str, help="Location of H5 dataset files.")
    parser.add_argument("--balanced_sampling", default=None, action="store_true", help="Whether to balance batches.")
    parser.add_argument("--cv", default=1, type=int, help="Number of CV folds.")
    parser.add_argument("--cv_idx", default=0, type=int, help="Specific CV fold index.")
    parser.add_argument("--eval_ratio", default=0.1, type=float, help="Ratio of validation subjects.")
    parser.add_argument("--max_eval_records", default=100, type=int, help="Max. number of subjects to use for eval.")
    parser.add_argument("--n_channels", default=None, type=int, help="Number of applied EEG channels.")
    parser.add_argument("--n_jobs", default=-1, type=int, help="Number of parallel jobs to run for prefetching.")
    parser.add_argument("--n_records", default=None, type=int, help="Total number of records to use.")
    parser.add_argument("--scaling", default="robust", choices=["robust", "standard"], help="How to scale EEG data.")
    parser.add_argument("--sequence_length", default=10, help="Number of sequences in each batch element.")
    parser.add_argument("--transforms", default=None, nargs="+", help="List of transforms to apply.")
    args = parser.parse_args()

    # Test dataset
    ds = EEGBCIDataset(**vars(args))
    print(ds)
    # next(iter(ds))

    # Test splitting of data
    train_data, eval_data = ds._split_data()
    assert len(train_data) > 0 and len(eval_data) > 0
    print(train_data)
    print(eval_data)
    print("Fin")
