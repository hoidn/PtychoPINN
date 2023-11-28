from setuptools import setup, find_packages
from functools import reduce

from setuptools import setup, find_packages

# Define the path to the data files within the package
package_data = {
    'ptycho': ['datasets/*.npz'],
}

setup(
    name='ptychoPINN',
    # ... other setup parameters ...
    package_data=package_data,
    include_package_data=True,
    # ... rest of the setup parameters ...

    packages=find_packages('.') + ['FRC'],
    package_dir={'ptychoPINN': 'ptycho', 'FRC': 'ptycho/FRC'},

    scripts = ['ptycho/train.py'],
    install_requires = [
        'protobuf==3.19.6',
        'dill==0.3.6',
        'numpy',
        'pandas==1.4.4',
        'pandas-datareader==0.10.0',
        'pathos==0.3.0',
        'scikit-learn==1.1.2',
        'scipy',
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
