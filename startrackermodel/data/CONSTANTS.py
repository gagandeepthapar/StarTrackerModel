"""
CONSTANTS 

List of constants to aid in development and maintain consistency

startrackermodel
"""
import logging
import logging.config
from numpy import pi as PI
import os

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            "datefmt": "%d/%b/%Y %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            # "stream": "ext://sys.stdout",  # Default is stderr
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        },
        "my.packg": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "__main__": {  # if __name__ == '__main__'
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# Set project level defaults
__CONST_ROOT = os.path.dirname(os.path.realpath(__file__))

# Useful files
YBSC_CSV = os.path.join(__CONST_ROOT, "YBSC.csv")
YBSC_PKL = os.path.join(__CONST_ROOT, "YBSC.pkl")

# Constants
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0
RAD2ARCSEC = 180.0 / PI * 3600

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

if __name__ == "__main__":
    logger.info("DEBUG TEST")
    logger.info(__CONST_ROOT)
