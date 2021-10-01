import argparse
import os

import mne

# from mne.datasets import eegbci


def fetch_subject(subject, path, runs=range(1, 15)):
    """Download a single subject from the EEGBCI dataset."""
    return mne.datasets.eegbci.load_data(subject, runs, path)


def download_eegbci(out_dataset_folder, n_first=None):
    """ Download the EEG Motor Movement/Imagery dataset (EEGBCI) from MNE."""

    # Make directory
    if not os.path.exists(out_dataset_folder):
        print(f"Creating output directory {out_dataset_folder}")
        os.makedirs(out_dataset_folder)

    # Get subjects
    for n in range(n_first):
        fetch_subject(n + 1, out_dataset_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory.\nWill be created if not available.",
    )
    parser.add_argument("-n", "--n_first", type=int, help="Number of recordings to download.")
    args = parser.parse_args()
    download_eegbci(args.output_dir, args.n_first)
