"""
Potential transforms for encoding models
"""
#IMPORT
##built-in
from typing import Dict,List
##third-party
import numpy as np
##local
from database_utils.functions import get_story_wordseqs, lanczosinterp2D

class ProcessEMA():
    """
    EMA processing transform - removes loudness information
    """
    def __init__(self):
        self.mask = np.ones(14, dtype=bool)
        self.mask[[12]] = False
    
    def __call__(self, sample:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Transform sample
        :param sample: dict, sample
        :return sample: dict, transformed sample
        """
        temp = sample['features']
        masked = temp[:, self.mask]
        norm_ema = np.empty((masked.shape), dtype=masked.dtype)
        for j in range(masked.shape[1]):
            jcol = masked[:,j]
            norm_ema[:,j] = 2 * np.divide((jcol - np.min(jcol)), (np.max(jcol)-np.min(jcol)))
        sample['features'] = norm_ema
        return sample

class ProcessPCA():
    """
    PCA processing transform for fitting encoding models with PCA features

    :param pcs: int, number of principle components in the PCA features that are being procesed
    """
    def __init__(self, pcs:int=13):
        self.pcs = pcs
        rand_matrix = np.random.random((self.pcs,self.pcs))
        eval, evec = np.linalg.eig(rand_matrix)
        self.r = evec
    
    def __call__(self, sample:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Transform sample
        :param sample: dict, sample
        :return sample: dict, transformed sample
        """
        vector = sample['features']
        rotated = np.matmul(vector, self.r)
        sample['features'] = rotated
        return sample

    
class Downsample():
    """
    Downsample features with lanczosinterp2D

    :param allstories: list, list of all story names
    :param window: int, windows for interpolation (default = 3)
    """
    def __init__(self, allstories:List, window:int=3):
        self.allstories = allstories
        self.window = window
        self.wordseqs = get_story_wordseqs(self.allstories)
    
    def __call__(self, sample:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Transform sample
        :param sample: dict, sample
        :return sample: dict, transformed sample
        """
        story = sample['story']
        vector = sample['features']
        times = sample['times']
        #Needs to be end time
        #after interpolation, plot time course related to the original uninterpolated feature
        #plot like in the speech model tutorial
        downsampled_vector = lanczosinterp2D(vector, times[:,1],
                                             self.wordseqs[story].tr_times, window=self.window)
        sample['features'] = downsampled_vector
        return sample
