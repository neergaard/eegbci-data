from setuptools import setup, find_packages

setup(
    name="eegbci",
    author="Alexander Neergaard Zahid, PhD",
    version="0.1.0",
    package_dir={"": "eegbci"},
    packages=find_packages(where="eegbci"),
    python_requires=">=3.9, <4",
    install_requires=["h5py", "mne", "scikit-learn"],
)
