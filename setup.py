# -*- coding: utf-8 -*-
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""


import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

short_description = 'Tinker Taylor py is a probabilistic decision-modelling framework for simulating and fitting the '
short_description += 'responses of people to probabilistic decision making tasks. '

setuptools.setup(
                name='Tinker Taylor py',

                # Versions should comply with PEP440.  For a discussion on single-sourcing
                # the version across setup.py and the project code, see
                # https://packaging.python.org/en/latest/single_source_version.html
                version='1.0.0a0',

                description=short_description,
                long_description=long_description,
                long_description_content_type="text/markdown",

                # The project's main homepage.
                url='https://github.com/djmhunt/TTpy',

                # Author details
                author='Dominic Hunt',
                author_email='d.hunt@gold.ac.uk',

                # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
                # Or https://pypi.org/classifiers/
                classifiers=[
                            'Development Status :: 5 - Production/Stable',
                            'Environment :: Console',
                            'Intended Audience :: Science/Research',
                            'License :: OSI Approved :: MIT License',
                            'Natural Language :: English',
                            'Operating System :: OS Independent',
                            'Programming Language :: Python :: 3',
                            'Topic :: Scientific/Engineering :: Medical Science Apps.',
                            'Topic :: Scientific/Engineering :: Artificial Life'
                            ],

                # What does your project relate to?
                keywords='neuroscience modelling Goldsmiths psychology PhD',

                # You can just specify the packages manually here if your project is
                # simple. Or you can use find_packages().
                packages=setuptools.find_packages(exclude=['contrib', 'doc', 'tests']),

                # List run-time dependencies here.  These will be installed by pip when
                # your project is installed. For an analysis of "install_requires" vs pip's
                # requirements files see:
                # https://packaging.python.org/en/latest/requirements.html
                install_requires='requirements.txt',
                python_requires='>=3.7',
)