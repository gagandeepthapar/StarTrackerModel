"""
composer

Interface with driver module to setup and run monte carlo

Steps to model:
    - Generate unique star tracker data from inputs
    - Generate unique environmental data from random distribution 
    - Determine visible stars and apply camera distortions 
    - Compare true VS measured quaternions 
    - Store results
    - Plot data

startrackermodel
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
