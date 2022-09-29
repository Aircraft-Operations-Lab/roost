[![DOI](https://zenodo.org/badge/542879252.svg)](https://zenodo.org/badge/latestdoi/542879252)

# roost: Robust Optimization Of Structured Trajectories

      
## What is roost?

The Python Library roost is a software package developed by UC3M. The roost is a tool for robust flight planning within the currently structured airspace. The main features of roost are: 1) integrates horizontal and vertical decision-making, 2) fast performance thanks to GPU-based parallelization, 3) considers climb, cruise, and descent phases
4) incorporate uncertainty in meteorological variables, as well as initial flight time and initial flight mass. 

**License:** roost is released under GNU Lesser General Public License v3.0 (LGPLv3). 

**Support:** Support of all general technical questions on roost, i.e., installation, application, and development, will be provided by Daniel González Arribas (dangonza@ing.uc3m.es) and Abolfazl Simorgh (abolfazl.simorgh@uc3m.es). 

**Core developer team:** Daniel González Arribas, Eduardo Andrés Enderiz, Abolfazl Simorgh, Manuel Soler. 

Copyright (C) 2022, Universidad Carlos III de Madrid

## How to run the library
The installation is the first step to working with roost. In the following, the steps required to install the library are provided.

0. It is highly recommended to create a virtual environment (e.g., roost):
```python
conda create -n env_roost
conda activate env_roost
```
1. Clone or download the repository. The roost source code is available on a public GitHub repository: XXX. The easiest way to obtain it is to clone the repository using git: git clone XXX.

2. Locate yourself in the roost (library folder) path, and run the following line, using terminal (MacOS and Linux) or cmd (Windows), which will install all dependencies:
```python
python setup.py install
```
it will install all required dependency.

## How to use it
There is a script in the roost (library folder) path, *main_run.py*, which provides a sample to get started with the library. This sample file contains some comments explaining the required inputs, problem configurations, selection of objective function (which includes flight planning objectives), optimization configurations, running, and output files. Notice that, in order to use roost for optimizing aircraft trajectories, the BADA4.0 license is required, and the script bada4.py needs to be included in the directory of the library.


