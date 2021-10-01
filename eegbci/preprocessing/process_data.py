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

from eegbci import fetch_subject


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
        raw_fnames = fetch_subject(subject, data_dir, runs=runs)
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


def run_pipeline(args):

    # Seed everything
    random.seed(42)
    np.random.seed(42)

    # Parse arguments
    data_dir = args.data_dir
    output_dir = args.output_dir
    fs = args.fs
    tmin, tmax = args.tmin, args.tmax
    subjects = args.subjects or range(109)
    freq_band = args.freq_band
    event_id = dict(rest=1, hands=2, feet=3)
    runs = list(range(3, 15))

    # Submit arguments to processing function
    process_subject = process_subject_fn(data_dir, fs, tmin, tmax, freq_band, event_id, runs)

    # Run over single subjects
    process_ok = 0
    for subject in subjects:
        logger.info(f"S{subject:03}")
        logger.info("---------------------------")
        raw, events, epochs = process_subject(subject)

        # Save subject data to disk
        out = write_h5(subject, epochs, output_dir)
        if not out:
            process_ok += 1

    return process_ok

    # logger.info(f"All subjects written to disk\n")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to EDF data.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Where to store H5 files.")
    parser.add_argument("--fs", type=int, default=128, help="Desired resampling frequency.")
    parser.add_argument("--tmin", type=float, default=-1., help="Start time of epochs relative to cue onset.")
    parser.add_argument("--tmax", type=float, default=4., help="End time of epochs relative to cue onset.")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help='Number of subjects to process. If None, all are processed.')
    parser.add_argument('--freq_band', type=float, nargs="+", default=[0.5, 35.], help='Lower and upper frequencies for passband filter.')
    args = parser.parse_args()
    # fmt: on

    if args.output_dir is None:
        args.output_dir = args.data_dir
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------------------------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == (len(vars(args)) - 1):
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    start = time()
    process_ok = run_pipeline(args)
    stop = time()
    logger.info(f"Preprocessing finished. {process_ok} subjects written to disk in {stop - start :.3f} seconds.")
