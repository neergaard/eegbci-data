import os

import mne


def fetch_subject(subject, path, runs=range(1, 15)):
    """Download a single subject from the EEGBCI dataset."""
    return mne.datasets.eegbci.load_data(subject, runs, path)


def fetch_eegbci(out_dataset_folder, n_first=109):
    """ Download the EEG Motor Movement/Imagery dataset (EEGBCI) from MNE."""

    # Make directory
    if not os.path.exists(out_dataset_folder):
        print(f"Creating output directory {out_dataset_folder}")
        os.makedirs(out_dataset_folder)

    # Get subjects
    for n in range(n_first):
        fetch_subject(n + 1, out_dataset_folder)
