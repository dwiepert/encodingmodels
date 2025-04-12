"""
Feature dataset

Author(s): Daniela Wiepert
Last modified: 04/11/2025
"""
#IMPORTS
##built-in
from pathlib import Path
from typing import Union,Dict,List
##third party
import numpy as np
from torch.utils.data import Dataset
import torchvision
##local
from ._load_feats import load_features
from ._transforms import *

class FeatureDataset(Dataset):
    """
    Custom feature dataset
    :param feature_type: str, name of feature for documentation purposes
    :param feature_dir: str/Path, root directory with feature files
    :param recursive: bool, boolean for whether to load features recursively (default = False)
    :param cci_features: cc interface (default = None)
    :param pre_transform: bool, indicate whether to run and transform all features in advance (otherwise transform done when each feature called)
    :param kwargs: additional arguments. Currently only 'pcs' is an argument of interest that is used when loading a pca-based feature set
    """
    def __init__(self, feature_type:str, feature_dir:Union[str,Path], recursive:bool=False, cci_features=None, pre_transform:bool=True, **kwargs):
        super().__init__()
        self.feature_type=feature_type
        self.root_dir = feature_dir
        self.cci_features = cci_features
        self.recursive=recursive
        print('Loading features...')
        self._load_data()
        self.stories = list(self.features.keys())
        self.transformed = False
        if self.feature_type=='ema':
            self.tf_list = [ProcessEMA()]
        elif 'pca' in self.feature_type:
            self.tf_list = [ProcessPCA(pcs=kwargs['pcs'])]
        else:
            self.tf_list = []

        self.tf_list.append(Downsample(self.stories))
        
        self.transforms = torchvision.transforms.Compose(self.tf_list)
        self.pre_transform = pre_transform
        if self.pre_transform:
            print('Transforming features...')
            self._transform_data()
    
    def _load_data(self):
        """
        Load features
        """
        self.features = load_features(feature_dir=self.root_dir, cci_features=self.cci_features,
                          recursive=self.recursive,ignore_str='times')
        self.times = load_features(feature_dir=self.root_dir, cci_features=self.cci_features,
                            recursive=self.recursive,search_str='times')
        #data = align_times(features, times)
    
    def __len__(self) -> int:
        """
        :return: int, length of data
        """
        return len(self.features)

    def __getitem__(self, idx:int) -> Dict[str,np.ndarray]:
        """
        Get item
        
        :param idx: int/List of ints/tensor of indices
        :return: dict, transformed sample
        """
        f = self.stories[idx]
        sample = {'story':f, 'features': self.features[f], 'times':self.times[f]}

        if not self.pre_transform:
            return self.transforms(sample)

        return sample
    
    def _transform_data(self) -> None:
        """
        Transform the entire dataset rather than one by one
        """
        new_features = {}

        for i in range(len(self.stories)):
            s = self.__getitem__(i)
            trans_s = self.transforms(s)
            new_features[self.stories[i]] = trans_s['features']
        self.features = new_features


    def get_feature_dict(self) -> Dict[str, np.ndarray]:
        """
        Get features

        :return self.features: feature dictionary
        """
        return self.features
    
    def get_story_dict(self, stories) -> Dict[str, np.ndarray]:
        """
        Get features for specific stories (still in dicitonary form)

        :return self.features: feature dictionary
        """
        storyd = {}
        for s in stories:
            storyd[s] = self.features[s]
        return storyd
    
    def get_stacked_features(self, stories) -> np.ndarray:
        """
        Get stacked features based off a list of stories
        """
        to_stack = []
        for s in stories:
            to_stack.append(self.features[s])
        stacked = np.vstack(to_stack)
        return stacked

        
        