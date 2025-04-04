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

You will also need to install [database_utils](https://github.com/dwiepert/database_utils)


## Initial feature extraction
This expects that features have already been extracted using [audio_features](https://github.com/dwiepert/audio_features) and are in a folder. Associated times need to exist somewhere, but the audio extraction will not always save out times. 

TODO: create a COPY OF TIMES WHENEVER IT DOESN'T SAVE OUT TIMES IN AUDIO FEATURES? THEN YOU NEVER NEED TO HAND IN TIMES... would be nice bro.

TODO:
DOWNSAMPLING: choose starting time for lanczos????
4. Encoding model stuff
5. Figureo out what train and test splits are
6. UPLOAD JSONS TO BUCKET TOO RATHER THAN REQUIRING A LOCAL PATH
7. Debug buckets


Download the data files and unzip in this directory. Should create a directory called data.

(If not using Anaconda) install dependencies: sudo apt-get update sudo apt-get install -y ipython ipython-notebook python-numpy python-scipy python-matplotlib cython python-pip python-pip python-dev python-h5py python-nibabel python-lxml python-shapely python-html5lib mayavi2 python-tables git

(If using Conda): conda install python 'cython=0.29.36' pytables h5py jupyter matplotlib numpy scipy (NOTE: some packages may be missing from this list)

(The cython requirement is from this issue: gallantlab/pycortex#490 (comment) )

Fetch and install pycortex: git clone https://github.com/gallantlab/pycortex.git cd pycortex; python setup.py install

Start a Jupyter notebook server in this directory (if you don't have one): jupyter notebook