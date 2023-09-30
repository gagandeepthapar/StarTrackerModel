"""
CONSTANTS 

List of constants to aid in development and maintain consistency

startrackermodel
"""

import numpy as np
import os
import logging

__CONST_ROOT = os.path.dirname(os.path.realpath(__file__))

# Useful files
YBSC = os.path.join(__CONST_ROOT, "YBSC.pkl")

# Constants
RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0
RAD2ARCSEC = 180.0 / np.pi * 3600

# Log levels
level_hash = {
    "DEBUG": logging.DEBUG,
    "D": logging.DEBUG,
    "INFO": logging.INFO,
    "I": logging.INFO,
    "WARNING": logging.WARNING,
    "W": logging.WARNING,
    "ERROR": logging.ERROR,
    "E": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "C": logging.CRITICAL,
}
