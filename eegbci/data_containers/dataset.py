import argparse
import logging
import os
import random
from itertools import compress
from typing import Callable, Optional, Tuple, List, Union

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
        subject_id = metadata["id"]
        record_id = os.path.basename(filename).split(".")[0]

        N, C, T = X.shape
        K = t.shape[-1]

        # Perform scaling
        if scaler:
            scaler.fit(X[:].transpose(1, 0, 2).reshape((C, N * T)).T)

        # Get class counts and locations of tasks
        class_counts = np.bincount(y[:].flatten(), minlength=K)  # y labels start from 1
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
        data_dir: str = "data/processed",
        balanced_sampling: bool = False,
        cv: int = 1,
        cv_idx: int = 0,
        eval_ratio: float = 0.1,
        max_eval_subjects: int = 100,
        n_channels: Optional[int] = None,
        n_jobs: Optional[int] = None,
        n_subjects: Optional[int] = None,
        n_runs: Optional[int] = None,
        subjects: Optional[list] = None,
        runs: Optional[list] = None,
        scaling: str = "robust",
        sequence_length: Union[int, str] = "full",
        transforms: Optional[List[Callable]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.balanced_sampling = balanced_sampling
        self.cv = cv
        self.cv_idx = cv_idx
        self.eval_ratio = eval_ratio
        self.max_eval_subjects = max_eval_subjects
        self.n_channels = n_channels
        self.n_jobs = n_jobs
        self.n_subjects = n_subjects
        self.n_runs = n_runs
        self.runs = runs
        self.scaling = scaling
        self.sequence_length = (
            sequence_length if isinstance(sequence_length, str) and sequence_length == "full" else int(sequence_length)
        )
        self.subjects = subjects
        self.transforms = transforms

        self.kf = KFold(n_splits=self.cv) if self.cv > 1 else None
        assert (n_subjects is None and subjects is not None) or (
            n_subjects is not None and subjects is None
        ), f"Please specify either the number or subjects, or provide a list of subjects to use, received: n_subjects={n_subjects}, subjects={subjects}"
        assert (n_runs is None and runs is not None) or (
            n_runs is not None and runs is None
        ), f"Please specify either the number or runs, or provide a list of runs to use, received: n_runs={n_runs}, runs={runs}"
        if self.n_subjects is not None:
            self.subjects = list(range(1, self.n_subjects + 1))
        elif self.subjects is not None:
            self.n_subjects = len(self.subjects)
        if self.n_runs is not None:
            self.runs = list(range(3, self.n_runs + 1))
        elif self.runs is not None:
            self.n_runs = len(self.runs)
        self.records = sorted(
            [
                f
                for f in os.listdir(self.data_dir)
                if (int(f[1:4]) in self.subjects) and (int(f.split(".")[0][-2:]) in self.runs)
            ]
        )

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
        # random.shuffle(self.records)
        random.shuffle(self.subjects)

    def _split_data(self) -> Tuple[Dataset, Dataset]:
        self._shuffle_records()
        if self.kf:
            self.train_idx, self.eval_idx = list(self.kf.split(range(self.n_subjects)))[self.cv_idx]
        else:
            n_eval = min(max(int(self.n_subjects * self.eval_ratio), 1), self.max_eval_subjects)
            self.train_idx = self.subjects[slice(n_eval, self.n_subjects)]
            self.eval_idx = self.subjects[slice(0, n_eval)]
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
\tData shape:               {self[0][0].shape}
\tEval ratio:               {self.eval_ratio}
\tNumber of channels:       {self.data_shape[0]}
\tNumber of subjects:       {self.n_subjects}
\tNumber of runs:           {self.n_runs}
\tNumber of sequences:      {len(self)}
\tScaling type:             {self.scaling}
\tSequence duration:        {int(self[0][0].shape[1] // self.metadata[self[0][2]]['fs'])} s
\tSequence length:          {self.sequence_length}
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
        dataset_group.add_argument("--max_eval_subjects", default=100, type=int, help="Max. number of subjects to use for eval.")
        dataset_group.add_argument("--n_channels", default=None, type=int, help="Number of applied EEG channels.")
        dataset_group.add_argument("--n_jobs", default=-1, type=int, help="Number of parallel jobs to run for prefetching.")
        dataset_group.add_argument("--n_subjects", default=None, type=int, help="Total number of subjects to use.")
        dataset_group.add_argument("--subjects", default=None, type=int, nargs='+', help="Specific subjects to use.")
        dataset_group.add_argument("--n_runs", default=None, type=int, help="Total number of runs to use for each subject.")
        dataset_group.add_argument("--runs", default=None, type=int, nargs='+', help="Specific runs to use for eaach subject.")
        dataset_group.add_argument("--scaling", default="robust", choices=["robust", "standard"], help="How to scale EEG data.")
        dataset_group.add_argument("--sequence_length", default=10, help="Number of sequences in each batch element.")
        # fmt: on
        return parser


class EEGBCISubset(Dataset):
    def __init__(self, dataset, subject_indices, balanced_sampling=False, name=None):
        self.dataset = dataset
        self.subject_indices = subject_indices
        self.balanced_sampling = balanced_sampling
        self.name = name
        self.subset_records = sorted(
            [r.split(".")[0] for r in self.dataset.records if int(r.split(".")[0][1:4]) in subject_indices]
        )  # [self.dataset.subjects[idx].split(".")[0] for idx in self.record_indices]
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

EEGBCI {self.name} subset
==============

Parameters:
------------------------------------------------------------------------------
\tBalanced sampling:        {self.dataset.balanced_sampling}
\tCross-validation folds:   {self.dataset.cv}
\tCurrent CV fold:          {self.dataset.cv_idx}
\tData directory:           {self.dataset.data_dir}
\tData shape:               {self[0][0].shape}
\tEval ratio:               {self.dataset.eval_ratio}
\tNumber of channels:       {self.dataset.data_shape[0]}
\tNumber of subjects:       {len(self.subject_indices)}
\tNumber of records:        {len(self)}
\tScaling type:             {self.dataset.scaling}
\tSequence duration:        {int(self[0][0].shape[1] // self.dataset.metadata[self[0][2]]['fs'])} s
\tSequence length:          {self.dataset.sequence_length if isinstance(self.dataset.sequence_length, str) else self.dataset.sequence_length}
\tSubset:                   {self.name}
==============================================================================
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser = EEGBCIDataset.add_dataset_specific_args(parser)
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
