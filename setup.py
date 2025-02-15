from setuptools import setup, find_packages
from encodingmodels._version import __version__

setup(
    name = 'encodingmodels.py',
    packages = find_packages(),
    author = 'HuthLab',
    python_requires='>=3.8',
    install_requires=[
        'numpy==1.26.4',
        'torchaudio==2.2.2',
        'torchvision==0.17.2',
        'torch==2.2.2',
        'tqdm==4.67.1',
        'cottoncandy==0.2.0',
        'scipy==1.15.0',
        'scikit-learn==1.6.1'
    ],
    include_package_data=False,  
    version = __version__,
)