"""
driver

Used for user-facing tool interaction.
Handles:
    - accepts arguments from different interfaces
    - unifies arguments 
    - passes into composer for data generation, computation

startrackermodel
"""
# pylint: disable=locally-disabled
import logging
import logging.config

from typing import Dict
import argparse
import json
import numpy as np
import pandas as pd
from time import perf_counter

from classes.plotter import Plotter
from classes.arguments import UserArguments
from scripts.composer import Composer
from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger("driver")

if __name__ == "__main__":
    # parse arguments
    args = UserArguments()

    # Create composer
    start_data_gen = perf_counter()  # start timer
    composer = Composer(args.sim_hw, args.sim_sw, args.sim_type)



    end_compute = perf_counter()
    logger.info("-")
    logger.info("Number of Runs [-]:\t\t%d", len_dataset)
    logger.info("Time to Simulate [sec]:\t%.5f", end_compute - start_data_gen)
    logger.info(
        "Average Sim Time [sec]:\t%.5f", (end_compute - start_data_gen) / len_dataset
    )
    logger.info(
        "Performance [asec]:\t\t%.3f +/- %.3f (1-sigma)",
        composer.model_data.ANGULAR_ERROR.mean(),
        composer.model_data.ANGULAR_ERROR.std(),
    )

    if args.show_plot:
        plotter = Plotter(dataset)
        plotter.standard()
        plotter.show()
