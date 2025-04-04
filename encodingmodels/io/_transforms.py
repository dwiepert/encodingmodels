"""
Potential transforms for encoding models
"""
#IMPORT
##built-in
from typing import Dict
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
    
class Downsample():
    """
    """
    def __init__(self, allstories, window:int=3):
        self.allstories = allstories
        self.window = window
        self.wordseqs = get_story_wordseqs(self.allstories)
    
    def __call__(self, sample):
        story = sample['story']
        vector = sample['features']
        times = sample['times']
        downsampled_vector = lanczosinterp2D(vector, times[:,0],
                                             self.wordseqs[story].tr_times, window=self.window)
        sample['features'] = downsampled_vector
        return sample
