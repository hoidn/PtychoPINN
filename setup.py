from setuptools import setup, find_packages
import os
from operator import add
from functools import reduce

#datafiles = reduce(add, [['../' + os.path.join(root, f) for f in files]
#    for root, dirs, files in os.walk('data/')])

setup(name = 'ptychoPINN',
#    packages = find_packages('.'),
#    package_dir = {'ptychoPINN': 'ptycho'},

    packages=find_packages('.') + ['FRC'],
    package_dir={'ptychoPINN': 'ptycho', 'FRC': 'ptycho/FRC'},

    scripts = ['ptycho/train.py'],
    #package_data = {'trader': ['data/*']},
    install_requires = [
        'protobuf==3.19.6',
        'dill==0.3.6',
        'numba==0.56.3',
        'numpy==1.23.0',
        'pandas==1.4.4',
        'pandas-datareader==0.10.0',
        'pathos==0.3.0',
        'scikit-learn==1.1.2',
        'scipy==1.9.1',
        'tensorboard==2.10.1',
        'tensorboard-data-server==0.6.1',
        'tensorboard-plugin-wit==1.8.1',
        'tensorflow==2.10.0',
        'tensorflow-datasets==4.6.0',
        'tensorflow-estimator==2.10.0',
        'tensorflow-hub==0.14.0',
        'tensorflow-metadata==1.10.0',
        'tensorflow-probability==0.18.0',
        'torchmetrics==0.9.3',
        'torchvision==0.13.1',
        'ujson',
        'matplotlib',
        'Pillow',
        'imageio',
        'ipywidgets',
        'tqdm',
        'tensorflow-addons'],
    zip_safe = False)
