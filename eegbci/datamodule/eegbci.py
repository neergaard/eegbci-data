import argparse

from sklearn import preprocessing
from pytorch_lightning import LightningDataModule

from eegbci import download_eegbci, process_eegbci
from eegbci.datamodule.dataset import EEGBCIDataset


class EEGBCIDataModule(LightningDataModule):
    def __init__(self, data_dir=None, subjects=None, processing_kwargs=None):
        super().__init__()
        self.data_dir = data_dir
        self.subjects = subjects
        self.processing_kwargs = processing_kwargs

    def prepare_data(self):
        if self.prepare_data:
            raw_dir = self.processing_kwargs["raw_dir"]
            fs = self.processing_kwargs["fs"]
            tmin = self.processing_kwargs["tmin"]
            tmax = self.processing_kwargs["tmax"]
            freq_band = self.processing_kwargs["freq_band"]
            download_eegbci(raw_dir, self.subjects)
            process_eegbci(raw_dir, self.data_dir, fs, tmin, tmax, self.subjects, freq_band)

    def setup(self, stage=None):

        # Partition data
        if stage == "fit" or stage is None:
            self.dataset = EEGBCIDataset(self.raw_dir)
            self.train_data, self.eval_data = self.dataset.split_data()

        if stage == "test":
            raise NotImplementedError

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", default="./data/processed", help="Location of H5 dataset files.")
    parser.add_argument("--raw_dir", default="./data", help="Location of raw EDF files.")
    parser = EEGBCI.add_argparse_args(parser)
    args = parser.parse_args()

    # Preprocessing arguments
    processing_kwargs = dict(raw_dir="./data", fs=128.0, tmin=-1.0, tmax=4.0, freq_band=[0.5, 35.0],)

    # Test datamodule
    eegbci = EEGBCIDataModule("data/processed", 12, processing_kwargs)
    eegbci.prepare_data()
