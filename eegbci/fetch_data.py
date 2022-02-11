import argparse
import logging
import sys

from .fetch import download_fns


logger = logging.getLogger("mne")


AVAILABLE_DATASETS = set(download_fns.keys())


def download_dataset(output_dir, n_first=None, cohort="eegbci"):
    download_fns[cohort](output_dir, n_first)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to output directory.\nWill be created if not available.",
    )
    parser.add_argument(
        "-c",
        "--cohort",
        type=str,
        help="Choice of EEG dataset (default 'eegbci').",
        default="eegbci",
        choices=AVAILABLE_DATASETS,
    )
    parser.add_argument("-n", "--n_first", default=109, type=int, help="Number of recordings to download.")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.log:
        file_handler = logging.FileHandler("logs/fetch_data.log", mode="w")
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

    download_dataset(args.output_dir, args.n_first, args.cohort)
