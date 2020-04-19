#!/bin/bash

# Install the required Python Modules
pip install -r requirements.txt

# Setup PyFlow for Optical Flow Computation
cd pyflow
python setup.py build_ext -ID
cd ..
