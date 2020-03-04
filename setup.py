# -*- coding: utf-8 -*-
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='Tinker Taylor py',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0.0-alpha',

    description='Tinker Taylor py is a probabilistic decision-modelling framework for simulating and fitting the responses of people to probabilistic decison making tasks. ',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/djmhunt/TTpy',

    # Author details
    author='Dominic Hunt',
    author_email='d.hunt@gold.ac.uk',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable'

        'Environment :: Console'

        'Intended Audience :: Science/Research'

        'License :: MIT'

        'Natural Language :: English'

        'Operating System :: OS Independent'

        'Programming Language :: Python :: 3.7'

        'Topic :: Scientific/Engineering :: Medical Science Apps'
    ],

    # What does your project relate to?
    keywords='neuroscience modelling Goldsmiths psychology PhD Dominic_Hunt',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'doc', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires='requirements.txt',

)


