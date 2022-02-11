import argparse
import logging
import os
import random
import sys
from time import time
from functools import partial

import h5py
import mne
import numpy as np
from sklearn.preprocessing import OneHotEncoder


COHORT = "eegbci"
logger = logging.getLogger("mne")


def process_subject_fn(subject, runs, data_dir, fs, tmin, tmax, freq_band, event_id):
    # def _process_subject(subject):
    # raw_fnames = fetch_subject(subject, data_dir, runs=runs)
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs, data_dir)
    raws_list = []
    epochs_list = []
    events_list = []
    for fname in raw_fnames:
        current_run = int(os.path.basename(fname).split(".")[0][-2:])
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)

        # Reference to average
        raw.set_eeg_reference("average", projection=True)

        # Extract events
        if current_run in [1, 2]:
            event_id_to_annotation = {"T0": 0, "T1": 1, "T2": 2}
        elif current_run in [3, 7, 11]:
            event_id_to_annotation = {"T0": 0, "T1": 1, "T2": 2}
        elif current_run in [4, 8, 12]:
            event_id_to_annotation = {"T0": 0, "T1": 5, "T2": 6}
        elif current_run in [5, 9, 13]:
            event_id_to_annotation = {"T0": 0, "T1": 3, "T2": 4}
        elif current_run in [6, 10, 14]:
            event_id_to_annotation = {"T0": 0, "T1": 7, "T2": 8}
        events, _ = mne.events_from_annotations(raw, event_id=event_id_to_annotation)

        # Resample and filter data
        raw, events = raw.resample(fs, events=events)
        iir_params = {"ftype": "butterworth", "order": 2, "output": "sos"}
        raw.filter(freq_band[0], freq_band[1], method="iir", iir_params=iir_params)

        # Epoch data
        # picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        if current_run in [1, 2]:
            epochs = mne.make_fixed_length_epochs(
                raw, duration=tmax - tmin, preload=True, proj=True, id=0, overlap=min((tmax - tmin) - 6.0, 0), verbose=False
            )
            epochs.event_id = event_id
        else:
            epochs = mne.Epochs(
                raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True, on_missing="ignore", verbose=False
            )
            epochs = epochs.crop(tmin=tmin, tmax=tmax, include_tmax=False)

        raws_list.append(raw)
        epochs_list.append(epochs)
        events_list.append(events)

    return raws_list, events_list, mne.concatenate_epochs(epochs_list)


# return _process_subject


def write_h5(subject, run_nr, epochs, output_dir):
    # Create subject ID and filename
    subject_id = f"S{subject:03}"
    run_id = f"R{run_nr:02}"
    filename = os.path.join(output_dir, subject_id + run_id + ".h5")
    os.makedirs(output_dir, exist_ok=True)

    # Create file
    with h5py.File(filename, "w") as f:

        # Write raw sensor data and labels in chunks
        logger.info(f"Writing signals and targets to file: {filename}")
        data = epochs.get_data()
        N, C, T = epochs.get_data().shape
        K = len(epochs.event_id)
        labels = epochs.events[:, -1].reshape(-1, 1)
        labels_onehot = OneHotEncoder(sparse=False, dtype=int, categories=np.arange(K)[np.newaxis]).fit_transform(
            labels
        )
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
        f.attrs["n_classes"] = K
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
    subjects_to_run = subjects if subjects is not None else range(1, 109 + 1)
    event_id = {
        "rest": 0,
        "hands/actual/left": 1,
        "hands/actual/right": 2,
        "hands/actual/both": 3,
        "feet/actual": 4,
        "hands/imagined/left": 5,
        "hands/imagined/right": 6,
        "hands/imagined/both": 7,
        "feet/imagined": 8,
    }
    runs = list(range(3, 15))  # We skip baselines

    # Submit arguments to processing function
    # process_subject = process_subject_fn(data_dir, fs, tmin, tmax, freq_band, event_id, runs)
    process_subject = partial(
        process_subject_fn, data_dir=data_dir, fs=fs, tmin=tmin, tmax=tmax, freq_band=freq_band, event_id=event_id
    )

    # Run over single subjects
    start = time()
    ok_subjects = 0
    for subject in subjects_to_run:
        logger.info(f"S{subject:03}")
        logger.info("---------------------------")
        process_ok = 0
        for run_nr in runs:
            raw, events, epochs = process_subject(subject=subject, runs=run_nr)

            # Save subject data to disk
            out = write_h5(subject, run_nr, epochs, save_dir)
            if not out:
                process_ok += 1

        if process_ok == len(runs):
            ok_subjects += 1

    stop = time()
    logger.info(f"Preprocessing finished. {ok_subjects} subjects written to disk in {stop - start :.3f} seconds.")
