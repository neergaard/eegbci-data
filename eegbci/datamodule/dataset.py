import argparse
import logging
import os

import numpy as np
from h5py import File
from joblib import Memory, delayed
from sklearn import preprocessing
from torch.utils.data import Dataset

from .utils import ParallelExecutor


logger = logging.getLogger("mne")
memory = Memory("data/.cache", mmap_mode="r", verbose=0)
SCALERS = {"robust": preprocessing.RobustScaler, "standard": preprocessing.StandardScaler}


@memory.cache
def _initialize_record(filename=None, scaling=None, sequence_length=None):
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

        index_to_record = [{"subject": subject_id, "idx": x} for x in range(sequence_counts)]

    return dict(
        metadata=(subject_id, metadata),
        scaler=(subject_id, scaler),
        class_indices=(subject_id, class_indices),
        class_counts=(subject_id, class_counts),
        sequence_counts=(subject_id, sequence_counts),
        index_to_record=index_to_record,
    )


class EEGBCIDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        balanced_sampling=None,
        cv_params=None,
        eval_ratio=None,
        max_eval_records=None,
        n_channels=None,
        n_jobs=None,
        n_records=None,
        scaling=None,
        sequence_length=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.balanced_sampling = balanced_sampling
        self.cv_params = cv_params
        self.eval_ratio = eval_ratio
        self.max_eval_records = max_eval_records
        self.n_channels = n_channels
        self.n_jobs = n_jobs
        self.n_records = n_records
        self.scaling = scaling
        self.sequence_length = (
            sequence_length if isinstance(sequence_length, str) and sequence_length == "full" else int(sequence_length)
        )
        self.records = sorted(os.listdir(self.data_dir))[: self.n_records]

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
        logger.info("Prefetching finished")
        self.class_counts = sum([s["class_counts"][1] for s in sorted_data])
        self.class_indices = dict([s["class_indices"] for s in sorted_data])
        self.index_to_record = [sub for s in sorted_data for sub in s["index_to_record"]]
        self.metadata = dict([s["metadata"] for s in sorted_data])
        self.scalers = dict([s["scaler"] for s in sorted_data])
        self.sequence_counts = dict([s["sequence_counts"] for s in sorted_data])

    def __len__(self):
        return sum(self.sequence_counts.values())

    def __getitem__(self, idx):

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
            current_subject = self.index_to_record[idx]["subject"]
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

        return x, t, current_subject, current_sequence


class EEGBCISubset(Dataset):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", default="./data/processed", type=str, help="Location of H5 dataset files.")
    parser.add_argument("--balanced_sampling", default=None, action="store_true", help="Whether to balance batches.")
    parser.add_argument("--n_cv", default=None, type=int, help="Number of CV folds.")
    parser.add_argument("--cv_idx", default=None, type=int, help="Specific CV fold index.")
    parser.add_argument("--eval_ratio", default=0.1, type=float, help="Ratio of validation subjects.")
    parser.add_argument("--max_eval_records", default=100, type=int, help="Max. number of subjects to use for eval.")
    parser.add_argument("--n_channels", default=None, type=int, help="Number of applied EEG channels.")
    parser.add_argument("--n_jobs", default=-1, type=int, help="Number of parallel jobs to run for prefetching.")
    parser.add_argument("--n_records", default=None, type=int, help="Total number of records to use.")
    parser.add_argument("--scaling", default="robust", choices=["robust", "standard"], help="How to scale EEG data.")
    parser.add_argument("--sequence_length", default=10, help="Number of sequences in each batch element.")
    args = parser.parse_args()
    args.cv_params = dict(cv_idx=args.cv_idx, n_cv=args.n_cv)
    del args.cv_idx, args.n_cv

    # Test dataset
    ds = EEGBCIDataset(**vars(args))
    print(len(ds))
    # next(iter(ds))
