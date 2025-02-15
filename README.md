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
1. Audio feature thing
2. Feature dataset compatible with all features 
3. Add in the downsampling stuff
4. Encoding model stuff