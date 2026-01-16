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
    package_dir={'ptychoPINN': 'ptycho', 'FRC': 'ptycho/FRC',
                 'autotest': 'ptycho/autotest'},

    scripts = ['ptycho/train.py'],
    install_requires = [
        'dill',
        'numpy',
        'pandas',
        'pandas-datareader',
        'pathos',
        'scikit-learn',
        'scipy==1.13.0',
        'tensorboard',
        'tensorboard-data-server',
        'tensorboard-plugin-wit',
        'tensorflow[and-cuda]',
        'keras==2.14.0',
        'tensorflow-datasets',
        'tensorflow-estimator',
        'tensorflow-hub',
        'tensorflow-probability==0.23.0',
        'ujson',
        'matplotlib',
        'Pillow',
        'imageio',
        'ipywidgets',
        'tqdm',
        'jupyter',
        'globus-compute-endpoint',
        'scikit-image',
        'opencv-python'
        ],
    zip_safe = False)
