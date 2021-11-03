# EEGBCI Data

This repository contains routines to download and preprocess data from the EEGBCI dataset [1] available from Physionet [2].
It also contains a PyTorch Dataset container for iterating over records, and a LightningDataModule for interfacing with Pytorch Lightning.

## Installation

Run the following to install the package into an environment:

```
git clone https://github.com/neergaard/eegbci-data.git
cd eegbci-data
pip install .
```

## Examples

### How to download data

This can be done either via command line like so:

```
python -m eegbci.fetch_data.download_data -o data/ \    # Save directory
                                          -n 10         # Ten subjects will be downloaded (if None, all)
```

or from within `python` by

```
from eegbci import download_eegbci

download_eegbci('data/', None)
```

The underlying download mechanisms are from the MNE toolbox, which will ask to confirm the default location of the dataset.

### How to run preprocessing

Again, this can be run using command line:

```
python -m eegbci.preprocessing.process_data -d data/ \              # Save directory from above
                                            -o data/processed/ \    # Output directory for processed files
                                            --fs 128 \              # Sampling frequency
                                            --tmin -1 \             # Start time of extracted epochs before cue onset
                                            --tmax 4 \              # End time of extracted epochs after cue onset
                                            --subjects None \       # Select specific subjects or select all (if None)
                                            --freq_band 0.3 35.     # Lower and upper frequencies for passband filter
```

or from within Python by

```
from eegbci import process_eegbci

process_eegbci(
    data_dir='data/',
    output_dir='data/processed',
    fs=128,
    tmin=-1,
    tmax=4,
    subjects=None,
    freq_band=[0.3, 35.]
)
```

### Run everything from within LightningDataModule

The provided LightningDataModule can handle both download and preprocessing of data:

```
import argparse

from eegbci import EEGBCIDataModule

parser = argparse.ArgumentParser()
parser = EEGBCIDataModule.add_argparse_args(parser)
args = parser.parse_args()

processing_kwargs = dict(
    raw_dir="./data",
    fs=128.0,
    tmin=-1.0,
    tmax=4.0,
    freq_band=[0.5, 35.0]
)
dm = EEGBCIDataModule(
    processing_kwargs=processing_kwargs,
    dataset_kwargs=**vars(args)
)
```

It is also possible to define CV splits and current CV index using supplied arguments, as well as control whether to balance samples in each batch.
A full list of arguments is available via `--help`:

```
python -m eegbci.datamodule.eegbci --help

usage: eegbci.py [-h] [--raw_dir RAW_DIR] [--overwrite] [-d DATA_DIR] [--balanced_sampling] [--cv CV] [--cv_idx CV_IDX] [--eval_ratio EVAL_RATIO] [--max_eval_records MAX_EVAL_RECORDS] [--n_channels N_CHANNELS] [--n_jobs N_JOBS] [--n_records N_RECORDS] [--scaling {robust,standard}]
                 [--sequence_length SEQUENCE_LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  --raw_dir RAW_DIR     Location of raw EDF files.
  --overwrite           Overwrite H5 files in directory.

dataset:
  -d DATA_DIR, --data_dir DATA_DIR
                        Location of H5 dataset files.
  --balanced_sampling   Whether to balance batches.
  --cv CV               Number of CV folds.
  --cv_idx CV_IDX       Specific CV fold index.
  --eval_ratio EVAL_RATIO
                        Ratio of validation subjects.
  --max_eval_records MAX_EVAL_RECORDS
                        Max. number of subjects to use for eval.
  --n_channels N_CHANNELS
                        Number of applied EEG channels.
  --n_jobs N_JOBS       Number of parallel jobs to run for prefetching.
  --n_records N_RECORDS
                        Total number of records to use.
  --scaling {robust,standard}
                        How to scale EEG data.
  --sequence_length SEQUENCE_LENGTH
                        Number of sequences in each batch element.
```

[1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

[2] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
