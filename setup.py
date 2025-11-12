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
        'imageio',
        'ipywidgets',
        'jupyter',
        'keras==2.14.0',
        'matplotlib',
        'numpy',
        'opencv-python',
        'pandas',
        'pandas-datareader',
        'pathos',
        'Pillow',
        'scikit-image',
        'scikit-learn',
        'scipy==1.13.0',
        'tensorboard',
        'tensorboard-data-server',
        'tensorboard-plugin-wit',
        'tensorflow[and-cuda]',
        'tensorflow-datasets',
        'tensorflow-estimator',
        'tensorflow-hub',
        'tensorflow-probability==0.23.0',
        'torch>=2.2',
        'tqdm',
        'ujson',
        'globus-compute-endpoint'
        ],
    extras_require = {
        'torch': [
            'lightning',     # PyTorch Lightning for training orchestration
            'mlflow',        # Experiment tracking
            'tensordict',    # Batch data handling
        ],
        
    },
    zip_safe = False)
