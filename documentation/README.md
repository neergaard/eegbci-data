# Data processing

The called functions are located in the eegbci-data repository, but the specific command line functions will be documented here.

## Fetching data

Assume the current directory is the project root, ie. `~/eegbci-data`, and that a correct virtual environment has been activated.

```
python -m eegbci.fetch_data -o data/ \
                            -c eegbci \
                            -n 109 \
                            --log
```

The `log` flag will save a log of the results in `logs/fetch_data.log`.

## Processing data

Assume the current directory is the project root, ie. `~/eegbci-data`, and that a correct virtual environment has been activated.

### No filtering or resampling

This will save an H5 file for each session for each subject.
The file contents will be chunked in 8 s windows: 2 s before cue onset, and 6 s after cue onset, meaning the 4 s task window is bookended by 2 s of rest.
This also means that each window is overlapping with the preceeding by 2 s.
The recordings are not resampled (original sampling rate of 160 Hz), and are not filtered either.

```
python -m eegbci.preprocess_data -d data/ \
                                 -o data/processed/unfiltered \
                                 -c eegbci \
                                 --tmin -2.0 \
                                 --tmax 6.0 \
                                 --log
```

### No filtering, no resampling, no overlap in segments

This will save an H5 file for each session for each subject.
The file contents will be chunked in 8 s windows: 2 s before cue onset, and 6 s after cue onset, meaning the 4 s task window is bookended by 2 s of rest.
This also means that each window is overlapping with the preceeding by 2 s.
The recordings are not resampled (original sampling rate of 160 Hz), and are not filtered either.

```
python -m eegbci.preprocess_data -d data/ \
                                 -o data/processed/unfiltered_no-overlap \
                                 -c eegbci \
                                 --tmin -1.0 \
                                 --tmax 5.0 \
                                 --log
```

### Filtering + resampling

This will save H5 files as above, but filtered between 0.3 Hz and 40 Hz.

```
python -m eegbci.preprocess_data -d data/ \
                                 -o data/processed/filtered_no-overlap \
                                 -c eegbci \
                                 --tmin -1.0 \
                                 --tmax 5.0 \
                                 --log \
                                 --freq_band 0.3 40.
```
