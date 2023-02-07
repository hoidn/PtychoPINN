from setuptools import setup, find_packages
import os
from operator import add
from functools import reduce

#datafiles = reduce(add, [['../' + os.path.join(root, f) for f in files]
#    for root, dirs, files in os.walk('data/')])

setup(name = 'ptychoPINN',
    packages = find_packages('.'),
    package_dir = {'ptychoPINN': 'ptycho'},
    scripts = ['ptycho/simtrain.py'],
    #package_data = {'trader': ['data/*']},
    install_requires = [],
    zip_safe = False)
