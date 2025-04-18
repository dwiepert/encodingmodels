"""
Fit and evaluate encoding models

Author(s): Daniela Wiepert, Aditya Vaidya, HuthLab
Last modified: 02/15/2025
"""
#IMPORTS
##built-in
import argparse
import json
import logging
from pathlib import Path
import warnings
##third-party
import torch
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cottoncandy as cc

##local
from encodingmodels.io import FeatureDataset
from encodingmodels.models import EncodingModel
from encodingmodels.utils import *

warnings.filterwarnings("ignore", category=UserWarning, module="cottoncandy") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir',type=Path,
                         help='Specify the path to features')
    parser.add_argument('--feature_type',type=str, required=True,
                         help='Specify the type of feature')
    parser.add_argument('--pcs', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--save_dir', type=Path, default=None,
                        help="Specify a local directory to save configuration files to. If not saving features to corral, this also specifies local directory to save files to.")
    parser.add_argument('--config', type=Path, help='Load arguments from a JSON file instead of setting them via command line')
    parser.add_argument("--recursive", action="store_true", help='Recursively find .wav,.flac,.npz files in the feature and stimulus dirs')
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--Rstories", nargs="+", default=None)
    parser.add_argument("--Pstories", nargs="+", default=None)
    parser.add_argument("--Pstories_trim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--chunk_sz", type=float, default=0.1)
    parser.add_argument("--context_sz", type=float, default=8.)
    cc_args = parser.add_argument_group('cc', 'cottoncandy related arguments (loading/saving to corral)')
    cc_args.add_argument('--feat_bucket', type=str, default=None, 
                         help="bucket with saved features")
    cc_args.add_argument('--stim_bucket', type=str, default='stimulidb', 
                         help="bucket with stimulus information")
    cc_args.add_argument('--out_bucket', type=str, default=None,
                         help="Bucket to save to.")
    model_args = parser.add_argument_group('model', 'model args')
    model_args.add_argument('--subject', type=str, required=True)
    model_args.add_argument('--trim', type=int, default=argparse.SUPPRESS)
    model_args.add_argument('--extra_trim', type=int, default=argparse.SUPPRESS)
    model_args.add_argument('--ndelays', type=int, default=argparse.SUPPRESS, help='Number of delays for FIR model. Each delay is offset by -2 starting at t=-2.')
    model_args.add_argument('--delays', nargs='+', type=int, default=[], help='Specify the exact time delays (as a list)')
    model_args.add_argument('--nboots', type=int, default=argparse.SUPPRESS)
    model_args.add_argument('--chunklen', type=int, default=argparse.SUPPRESS)
    model_args.add_argument('--nchunks', default=125, help='absolute number of chunks if an int. If a float, a proportion of training TRs (0.25 is a good number)')
    model_args.add_argument('--singcutoff', type=float, default=argparse.SUPPRESS)
    model_args.add_argument('--use_corr', action='store_true')
    model_args.add_argument('--single_alpha', action='store_true')
    model_args.add_argument('--save_weights', action='store_true')
    model_args.add_argument('--alphas_logspace', default=argparse.SUPPRESS, type=int_tuple, help='A tuple of '
						'ints, describing the inputs to np.logspace, which '
						'generates the list of possible ridge parameters.')
    model_args.add_argument('--save_pred', action='store_true')
    model_args.add_argument('--save_crossval', action='store_true')
    model_args.add_argument('--overwrite', action='store_true')
    model_args.add_argument('--nuisance', type=str, default=argparse.SUPPRESS)
    model_args.add_argument('--ignore_dialogue', action='store_true',
                        help='ignore some specific stories')
    model_args.add_argument('--scaling_story_splits', action='store_true',
                        help='ignore some specific stories, and add more stories to test set')
    args = parser.parse_args()
    
    ##LOAD FROM CONFIG IF EXISTS
    args_dict = {}
    if args.config:
        with open(args.config, 'r') as f:
            args_dict.update(json.load(f))
            for key in ['ignore_dialogue', 'scaling_story_splits']:
                if key in args_dict:
                    raise NotImplementedError("flag settings are not supported yet via config file")

    args_dict.update(args.__dict__)
    if args_dict['save_dir'] is None:
        assert args_dict['out_bucket'] is not None
    else:
        args_dict['save_dir'].mkdir(parents=True, exist_ok=True)

    ## FEATURE DATASET KWARGS
    featdb_kwargs = {k: v for k, v in args_dict.items() if k in ['feature_dir','feature_type', 'recursive', 'pcs']}
    if 'pca' in featdb_kwargs['feature_type']: assert 'pcs' in featdb_kwargs
        
    if args_dict['feat_bucket'] is not None:
        cci_features = cc.get_interface(args_dict['feat_bucket'], verbose=False)
        print("Loading features from bucket", cci_features.bucket_name)
    else:
        cci_features = None
        print('Loading features from local filesystem.')
    featdb_kwargs['cci_features'] = cci_features 
    featdb_kwargs['pre_transform'] = True

    
    ## DETERMINE RSTORIES AND PSTORIES
    if args.sessions is None:
        assert (args_dict['Rstories'] is not None) or (args_dict['Pstories']) is not None
    else:
        assert args.stim_bucket is not None
        all_stories, train_stories, test_stories = get_stories_in_sessions(args.sessions, args.stim_bucket)
        args_dict['Rstories'] = train_stories
        args_dict['Pstories'] = test_stories

    ## ENCODING MODEL KWARGS
    encmodel_kwargs = {k: v for k, v in args_dict.items() if k not in ['feature_dir', 'recursive', 'feat_bucket', 'stim_bucket', 'jobs', 'config', 'pcs']}

    nchunks = encmodel_kwargs['nchunks']
    if str_is_type(nchunks, int):
        encmodel_kwargs['nchunks'] = int(nchunks)
    elif str_is_type(nchunks, float):
        encmodel_kwargs['nchunks'] = float(nchunks)
    else:
        raise ValueError(f"nchunks must be either an int or float, but received {encmodel_kwargs['nchunks']}")

    ## LOGGING
    logging.getLogger("counter").setLevel(logging.INFO)

    ## Set up Dataset
    feat_dataset = FeatureDataset(**featdb_kwargs)

    ## Get features for training
    feats = feat_dataset.get_feature_dict()
    encmodel_kwargs['feats'] = feats

    ## Set up model
    model = EncodingModel(**encmodel_kwargs)
    #TODO: SAVING AND FINDING FROM A LOCAL DIR
    func = None
    if model.cci is not None:
        func = model.cci.exists_object

    results_found = recursive_dict_map(model.result_paths, func=func)
    print(results_found)
    if (not model.new_config) and recursive_dict_all(results_found):
        if args_dict['overwrite']:
            print('Overwriting ridge parameters.')
        else:
            print('Ridge results exist. To overwrite, set boolean to True.')
            exit(0)
    model.run_regression()
    