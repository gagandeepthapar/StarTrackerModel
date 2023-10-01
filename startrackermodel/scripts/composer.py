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

import logging
import logging.config
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from classes import arguments as args

from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def create_data():
    return


def compose_arguments():
    return


if __name__ == "__main__":
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100) * 5 + 3
