# Encoding models
New encoding models repo that is compatible with the array of features extracted from [audio_features](https://github.com/dwiepert/audio_features).

## Setup
In order to install all components easily, it is best to have a conda environment (some packages installed with conda and not setup.py)

To install, use

```
$ git clone https://github.com/dwiepert/encodingmodels.git
$ cd audio_features
$ pip install . 
```

You will also need to install:
* [database_utils](https://github.com/dwiepert/database_utils)
* pytables, install with `conda install pytables`


## Initial feature extraction
This expects that features have already been extracted using [audio_features](https://github.com/dwiepert/audio_features) and are in a folder with the corresponding times files. 

## FeatureDataset
Features are loaded in using the [FeatureDataset](https://github.com/dwiepert/encodingmodels/encodingmodels/io/_dataset.py) class. 

This dataset requires the following parameters:
* `feature_type`: string name of the kind of features included in the dataset. This is important for initializing data processing transforms. 'ema' features will be processed with [ProcessEMA](https://github.com/dwiepert/encodingmodels/encodingmodels/io/_transforms.py) and features with 'pca' in the type will be processed with [ProcessPCA](https://github.com/dwiepert/encodingmodels/encodingmodels/io/_transforms.py)
* `feature_dir`: path-like object containing the root directory with feature files

There are additional default parameters that will likely be of interest:
* `recursive`: boolean, set True if the features files are stored in multiple subdirectories (default=False)
* `cci_features`: pass a cotton candy interface object if working with features stored in coral (default=None)
* `pre_transform`: boolean, set True if features should be transformed upon initialization of the dataset (default = True)

NOTE: If the features are PCA-based, an additional parameters `pcs` is required that contains the number of principle components used to form the features (int)

This Dataset can be used like a traditional torch Dataset, or you can get the features out as either a dictionary mapping stories to the corresponding features (`get_feature_dict` for all stories, `get_story_dict(stories)` for a subset of stories) or stacked features for a subset of stories (`get_stacked_features(stories)`).

## EncodingModel
The [EncodingModel](https://github.com/dwiepert/encodingmodels/encodingmodels/models/_encoding.py) class handles fitting encoding models from start to finish. All that is required is to initialize the EncodingModel and then call `model.run_regression()` with no parameters.

The EncodingModel class is initialized with the following required parameters:
* `subject`: str, fMRI scan subject name
* `out_bucket`: str/None, name of s3 bucket where outputs can be saved
* `feats`:dict, dictionary of features mapping string story name to the loaded numpy array
* `feature_type`: str, name of feature
* `sessions`: str list, list of session numbers to get fMRI responses from (as strings)
Parameters of interest with default values are:
* `Rstories`: List of stories to use for training - if not specified, will find them using `sessions`
* `Pstories`:  List of stories to use for testing - if not specified, will find them using `sessions`
* `save_dir`:str/Path, optional local save path, must be give if out_bucket is None
* `chunk_sz`: float, feature chunk (from feature extraction) in seconds (default = 0.1) *dependent on audio_features parameters
* `context_sz`: *float, context size used for feature extraction in seconds (default = 8.0)
* `save_weights`: bool, save out model weights (default = True)
* `overwrite`: bool, whether to overwrite saved files when running (default = False)
* `save_crossval`: bool, save out cross validation results (default = True)
* `save_pred`:bool, save out model predictions (default = True)
Check out the __init__ function of EncodingModel for additional default parameters. 

## Running with run_encodingmodels.py
[run_encodingmodels.py](https://github.com/dwiepert/encodingmodels/run_encodingmodels.py)

This script will load features and fit models in one place. 

The main parameters are as follows:
* `--subject`: sepcify the scanning subject
* `--feature_dir/--feat_bucket`: specify either directory or bucket where features are stored
* `--feature_type`: specify the feature type
* `--save_dir/--out_bucket`: specify either the directory or bucket where results should be saved to
* `--recursive`: boolean indicating whether features files are stored in subdirectories
* `--sessions`: list of scanning sessions to include
    * you can also give `--Rstories` and `--Pstories` if you want to use a subset of stories from the given sessions
* toggle `--save_weight`, `--save_pred`, `--save_crossval`
* optionally toggle `--overwrite`
* optionally change `--nboots` from default value of 10 (higher=more stable, 50 can be a good option)
* If features are extracted with different `context_sz` or `chunk_sz`, set those here (IN SECONDS)

There are a number of suppressed parameters/default parameters. Please see the script for those additional parameters.

An example command would look like:
```python run_encodingmodels.py --subject=<SUBJECT_NAME> --feature_dir=<PATH_TO_DIR> --feature_type=<FEATURE_NAME> --save_dir=<PATH_TO_DIR> --sessions 1 2 3 4 5 --nboots=10 --save_weights --save_pred --save_crossval ```