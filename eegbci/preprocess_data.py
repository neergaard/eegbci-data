import argparse
import logging
import sys

from .preprocessing import preprocessing_fns


logger = logging.getLogger("mne")


AVAILABLE_DATASETS = set(preprocessing_fns.keys())


def preprocess_data(args):
    preprocessing_fns[args.cohort](**vars(args))


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to EDF data.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Where to store H5 files.")
    parser.add_argument('-c', '--cohort', type=str, default='eegbci', help="Choice of preprocessing function (default 'eegbci').")
    parser.add_argument("--fs", type=int, default=128, help="Desired resampling frequency.")
    parser.add_argument("--tmin", type=float, default=-1., help="Start time of epochs relative to cue onset.")
    parser.add_argument("--tmax", type=float, default=4., help="End time of epochs relative to cue onset.")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help='Number of subjects to process. If None, all are processed.')
    parser.add_argument('--freq_band', type=float, nargs="+", default=[0.5, 35.], help='Lower and upper frequencies for passband filter.')
    args = parser.parse_args()
    # fmt: on

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------------------------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == (len(vars(args)) - 1):
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    preprocess_data(args)
