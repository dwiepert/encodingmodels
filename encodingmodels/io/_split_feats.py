"""
Split features 

FROM audio_features 
Author(s): Daniela Wiepert 
Last modified: 11/28/2024
"""
#IMPORTS
##built-in
import os
from pathlib import Path 
from typing import Dict
##third-party
import numpy as np

def split_features(features:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    """
    Split feature dict
    :param features: dict, dictionary of features {path: feature vectors}
    :return new_feat_dict: dict, dictionary of split features {stimulus name: feature vectors}
    """
    path_list = features['path_list']
    path_to_fname = {}
    new_feat_dict = {}

    for p in path_list:
        fname = os.path.splitext(Path(p).name)
        fname = fname[0].split(sep="_")[0]
        path_to_fname[str(p)] = fname
    
    for f in features:
        if f != 'path_list':
            feats = features[f]
            fname = path_to_fname[str(f)]
            new_feat_dict[fname] = feats
    
    return new_feat_dict


