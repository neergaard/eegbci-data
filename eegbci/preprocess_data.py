import argparse
import logging
import sys

from .preprocessing import preprocessing_fns


logger = logging.getLogger("eegbci")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


AVAILABLE_DATASETS = set(preprocessing_fns.keys())


def process_data(cohort="eegbci", *args, **kwargs):
    preprocessing_fns[cohort](**kwargs)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to EDF data.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Where to store H5 files.")
    parser.add_argument('-c', '--cohort', type=str, default='eegbci', help="Choice of preprocessing function (default 'eegbci').")
    parser.add_argument("--fs", type=int, default=None, help="Desired resampling frequency.")
    parser.add_argument("--tmin", type=float, default=0., help="Start time of epochs relative to cue onset.")
    parser.add_argument("--tmax", type=float, default=4., help="End time of epochs relative to cue onset.")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help='Number of subjects to process. If None, all are processed.')
    parser.add_argument('--freq_band', type=float, nargs="+", default=None, help='Lower and upper frequencies for passband filter.')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()
    # fmt: on

    if args.log:
        file_handler = logging.FileHandler("logs/process_data.log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------------------------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == (len(vars(args)) - 1):
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    if args.log:
        args = vars(args)
        args.pop("log")

    process_data(**args)
