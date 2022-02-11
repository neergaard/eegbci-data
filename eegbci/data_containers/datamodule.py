import argparse
import os
from typing import Optional

from pytorch_lightning import LightningDataModule

from eegbci.fetch import fetch_eegbci
from eegbci.preprocessing import preprocess_eegbci
from eegbci.data_containers.base_datamodule import BaseDataModule
from eegbci.data_containers.dataset import EEGBCIDataset
from eegbci.data_containers.mixins import CollateFnMixin
from eegbci.data_containers.mixins import DataloaderMixin


class EEGBCIDataModule(DataloaderMixin, CollateFnMixin, LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        processing_kwargs: Optional[dict] = None,
        dataset_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        self.processing_kwargs = processing_kwargs

    @property
    def sampling_frequency(self):
        assert self._sampling_frequency is not None, f"Specify sampling_frequency property for the DataModule ({self})"
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency: int) -> int:
        self._sampling_frequency = sampling_frequency

    def prepare_data(self):
        data_dir = self.dataset_kwargs["data_dir"]
        if self.processing_kwargs is not None:
            overwrite = self.processing_kwargs["overwrite"]
            if overwrite or not any([p.endswith(".h5") for p in os.listdir(data_dir)]):
                raw_dir = self.processing_kwargs["raw_dir"]
                fs = self.processing_kwargs["fs"]
                tmin = self.processing_kwargs["tmin"]
                tmax = self.processing_kwargs["tmax"]
                freq_band = self.processing_kwargs["freq_band"]
                n_records = self.dataset_kwargs["n_records"]
                fetch_eegbci(raw_dir, n_records)
                preprocess_eegbci(raw_dir, data_dir, fs, tmin, tmax, n_records, freq_band)

    def setup(self, stage=None):

        # Partition data
        if stage == "fit" or stage is None:
            self.dataset = EEGBCIDataset(**self.dataset_kwargs)
            self.train_data, self.eval_data = self.dataset._split_data()

        if stage == "test":
            self.test_data = self.eval_data

        self.sampling_frequency = self.dataset.metadata[self.dataset.records[0].split(".")[0]]["fs"]

    @staticmethod
    def add_argparse_args(parent_parser):
        # fmt: off
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        dataset_group = parser.add_argument_group("datamodule")
        dataset_group.add_argument("-bs", "--batch_size", default=8, type=int, help="Batch size.")
        dataset_group.add_argument("--num_workers", default=0, help="Number of workers to use in the DataLoader.")
        # fmt: on
        return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--raw_dir", default="./data", help="Location of raw EDF files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite H5 files in directory.")
    parser = EEGBCIDataModule.add_argparse_args(parser)
    parser = EEGBCIDataset.add_dataset_specific_args(parser)
    args = parser.parse_args()

    # Preprocessing arguments
    processing_kwargs = dict(raw_dir="./data", fs=128.0, tmin=-0.0, tmax=4.0, freq_band=[0.5, 35.0], overwrite=args.overwrite)

    # Test datamodule
    dm = EEGBCIDataModule(processing_kwargs=processing_kwargs, **vars(args))
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    print(dm.train_data)
    print(dm.eval_data)
    print(dm.test_data)

    train_loader = dm.train_dataloader()
    print(next(iter(train_loader)))
