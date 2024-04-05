from setuptools import setup, find_packages

setup(
    name='ptychoPINN',
    # ... other setup parameters ...
    package_data={
        # Specify the correct path to the data directory at the top-level
        '': ['datasets/*.npz'],
    },
    include_package_data=True,
    # ... rest of the setup parameters ...

    packages=find_packages('.') + ['FRC'],
    package_dir={'ptychoPINN': 'ptycho', 'FRC': 'ptycho/FRC'},

    scripts = ['ptycho/train.py'],
    install_requires = [
        'dill',
        'numpy',
        'pandas',
        'pandas-datareader',
        'pathos',
        'scikit-learn',
        'scipy',
        'tensorboard',
        'tensorboard-data-server',
        'tensorboard-plugin-wit',
        'tensorflow[and-cuda]',
        'keras==2.14.0',
        'tensorflow-datasets',
        'tensorflow-estimator',
        'tensorflow-hub',
        'tensorflow-probability',
        'ujson',
        'matplotlib',
        'Pillow',
        'imageio',
        'ipywidgets',
        'tqdm',
        'tensorflow-addons',
        'jupyter',
        'scikit-image',
        'opencv-python'
        ],
    zip_safe = False)
