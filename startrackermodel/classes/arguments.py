"""
arguments

Class to unify user-arguments from different spaces into singular interface

startrackermodel
"""

import logging
import logging.config
import json

from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class UserArguments:
    def __init__(self):
        logger.critical("Creating User Arguments")
        return
