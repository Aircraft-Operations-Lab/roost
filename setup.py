#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='roost',
      version='0.1',
      description='Robust Optimization Of Structured Trajectories',
      license='lgpl-3.0',
      author='Daniel González Arribas, Eduardo Andrés Enderiz, Abolfazl Simorgh, Manuel Soler',
      author_email='dangonza@ing.uc3m.es',
      packages=['roost'],
      keywords=['Aircraft trajectory optimization', 'Meteorological forecast uncertainty', 'Robustness', 'Structured airspace' ,'Climate Impacts of Aviation', 'Algorithmic Climate Change Functions'],
      install_requires=[
          'networkx',
          'jinja2',
          'pandas', 'numpy', 'pycuda', 'roc3', 'matplotlib', 'mitos',
      ],
      include_package_data=True,
      zip_safe=False,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',],)