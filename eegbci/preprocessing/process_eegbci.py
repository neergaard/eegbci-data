import argparse
import logging
import os
import random
import sys
from time import time

import h5py
import mne
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# from eegbci.data_download.fetch_data import fetch_subject


COHORT = "eegbci"
logger = logging.getLogger("mne")
# ch = logging.StreamHandler()
# ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%I:%M:%S"))
# logger.addHandler(ch)
# logging.basicConfig(
#     format="%(asctime)s | %(message)s", datefmt="%I:%M:%S",
# )


def process_subject_fn(data_dir, fs, tmin, tmax, freq_band, event_id, runs):
    def _process_subject(subject):
        # raw_fnames = fetch_subject(subject, data_dir, runs=runs)
        raw_fnames = mne.datasets.eegbci.load_data(subject, runs, data_dir)
        raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames])
        mne.datasets.eegbci.standardize(raw)  # set channel names
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # strip channel names of "." characters
        raw.rename_channels(lambda x: x.strip("."))

        # Extract events
        events, _ = mne.events_from_annotations(raw, event_id=dict(T0=1, T1=2, T2=3))

        # Resample data
        raw, events = raw.resample(fs, events=events)

        # Apply band-pass filter
        iir_params = {"ftype": "butterworth", "order": 2, "output": "sos"}
        raw.filter(freq_band[0], freq_band[1], method="iir", iir_params=iir_params)

        # Epoch data
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
        epochs = epochs.crop(tmin=tmin, tmax=tmax, include_tmax=False)

        return raw, events, epochs

    return _process_subject


def write_h5(subject, epochs, output_dir):
    # Create subject ID and filename
    subject_id = f"S{subject:03}"
    filename = os.path.join(output_dir, subject_id + ".h5")
    os.makedirs(output_dir, exist_ok=True)

    # Create file
    with h5py.File(filename, "w") as f:

        # Write raw sensor data and labels in chunks
        logger.info(f"Writing signals and targets to file: {filename}")
        data = epochs.get_data()
        labels = epochs.events[:, -1].reshape(-1, 1)
        labels_onehot = OneHotEncoder(sparse=False, dtype=int).fit_transform(labels)
        N, C, T = data.shape
        _, K = labels_onehot.shape
        f.create_dataset("signals", (N, C, T), data=data, chunks=(1, C, T))
        f.create_dataset("targets/labels", (N, 1), data=labels, chunks=(1, 1))
        f.create_dataset("targets/onehot", (N, K), data=labels_onehot, chunks=(1, K))

        # Enter metadata
        logger.info(f"Writing metadata to file: {filename}\n")
        f.attrs["ch_names"] = epochs.info["ch_names"]
        f.attrs["data_shape"] = epochs.get_data().shape
        f.attrs["event_id"] = tuple(epochs.event_id.values())
        f.attrs["event_categories"] = tuple(epochs.event_id.keys())
        f.attrs["id"] = subject_id
        f.attrs["fs"] = epochs.info["sfreq"]
        f.attrs["n_channels"] = C
        f.attrs["n_events"] = [sum(epochs.events[:, -1] == v) for k, v in epochs.event_id.items()]
        f.attrs["n_epochs"] = N
        f.attrs["n_timepoints"] = T
        f.attrs["timepoints"] = epochs.times


def preprocess_eegbci(data_dir, output_dir=None, fs=128, tmin=-1.0, tmax=4.0, subjects=None, freq_band=[0.5, 35.0]):

    # Seed everything
    random.seed(42)
    np.random.seed(42)

    # Parse arguments
    save_dir = output_dir or os.path.join(data_dir, "processed")
    subjects_to_run = subjects or range(1, 109 + 1)
    event_id = dict(rest=1, hands=2, feet=3)
    runs = list(range(3, 15))

    # Submit arguments to processing function
    process_subject = process_subject_fn(data_dir, fs, tmin, tmax, freq_band, event_id, runs)

    # Run over single subjects
    start = time()
    process_ok = 0
    for subject in range(1, subjects_to_run + 1):
        logger.info(f"S{subject:03}")
        logger.info("---------------------------")
        raw, events, epochs = process_subject(subject)

        # Save subject data to disk
        out = write_h5(subject, epochs, save_dir)
        if not out:
            process_ok += 1

    stop = time()
    logger.info(f"Preprocessing finished. {process_ok} subjects written to disk in {stop - start :.3f} seconds.")

    # logger.info(f"All subjects written to disk\n")
