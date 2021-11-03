import argparse
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from eegbci import download_eegbci, process_eegbci
from eegbci.data_containers.dataset import EEGBCIDataset


class EEGBCIDataModule(LightningDataModule):
    def __init__(self, processing_kwargs=None, **dataset_kwargs):
        super().__init__()
        self.dataset_kwargs = dataset_kwargs
        self.processing_kwargs = processing_kwargs

    def prepare_data(self):
        raw_dir = self.processing_kwargs["raw_dir"]
        fs = self.processing_kwargs["fs"]
        tmin = self.processing_kwargs["tmin"]
        tmax = self.processing_kwargs["tmax"]
        freq_band = self.processing_kwargs["freq_band"]
        n_records = self.dataset_kwargs["n_records"]
        data_dir = self.dataset_kwargs["data_dir"]
        overwrite = self.processing_kwargs["overwrite"]
        download_eegbci(raw_dir, n_records)
        if overwrite or not any([p.endswith(".h5") for p in os.listdir(data_dir)]):
            process_eegbci(raw_dir, data_dir, fs, tmin, tmax, n_records, freq_band)

    def setup(self, stage=None):

        # Partition data
        if stage == "fit" or stage is None:
            self.dataset = EEGBCIDataset(**self.dataset_kwargs)
            self.train_data, self.eval_data = self.dataset._split_data()

        if stage == "test":
            self.test_data = self.eval_data

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_data.batch_size,
            shuffle=True,
            num_workers=self.train_data.n_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.eval_data.batch_size,
            shuffle=False,
            num_workers=self.eval_data.n_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.eval_data.batch_size,
            shuffle=False,
            num_workers=self.eval_data.n_workers,
            pin_memory=True,
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        return EEGBCIDataset.add_dataset_specific_args(parent_parser)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--raw_dir", default="./data", help="Location of raw EDF files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite H5 files in directory.")
    parser = EEGBCIDataset.add_dataset_specific_args(parser)
    args = parser.parse_args()

    # Preprocessing arguments
    processing_kwargs = dict(
        raw_dir="./data", fs=128.0, tmin=-1.0, tmax=4.0, freq_band=[0.5, 35.0], overwrite=args.overwrite
    )

    # Test datamodule
    eegbci = EEGBCIDataModule(processing_kwargs=processing_kwargs, **vars(args))
    eegbci.prepare_data()
    eegbci.setup("fit")
    eegbci.setup("test")

    print(eegbci.train_data)
    print(eegbci.eval_data)
    print(eegbci.test_data)