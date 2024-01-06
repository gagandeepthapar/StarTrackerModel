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
from time import perf_counter
import signal

import numpy as np
import pandas as pd

from classes.plotter import Plotter
from classes.arguments import UserArguments
from scripts.composer import Composer
from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger("driver")

INTERRUPTED = False


def signal_handler(sig, frame):
    "Set global flag to stop compute"
    logger.critical("Ending Simulation at End of Current Loop")
    global INTERRUPTED
    INTERRUPTED = True


if __name__ == "__main__":
    # set signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # parse arguments
    args = UserArguments()

    # Create composer
    start_data_gen = perf_counter()  # start timer
    composer = Composer(args.sim_hw, args.sim_sw, args.sensor, args.env, args.sim_type)

    # initialize metrics
    dataset = pd.DataFrame()
    dataset_std_ratio, prev_dataset_std, batch_num = 1, 1, 1
    len_dataset = 0
    while dataset_std_ratio > args.eps and len_dataset <= args.max_runs:
        batch_start = perf_counter()

        # generate and run model
        composer.generate_data(args.num_runs, args.param_select)  # generate data
        composer.run_model()  # run stochastic model
        batch_end = perf_counter()

        # compute metrics
        dataset = pd.concat([dataset, composer.model_data], axis=0, ignore_index=True)
        len_dataset = len(dataset.index)

        # compute stopping criterion
        dataset_std_ratio = (
            np.abs(np.std((dataset.ANGULAR_ERROR)) - prev_dataset_std)
        ) / prev_dataset_std
        prev_dataset_std = np.std((dataset.ANGULAR_ERROR))

        # update terminal
        logger.info(
            "Batch\t%d (%d Scenes): %.3f +/- %.3f (%.5f STD Change); %.5f Run Time",
            batch_num,
            len_dataset,
            np.mean((dataset.ANGULAR_ERROR)),
            prev_dataset_std,
            dataset_std_ratio,
            batch_end - batch_start,
        )
        logger.critical(
            np.sqrt(
                np.square(dataset.ANGULAR_ERROR.to_numpy()).sum() / len(dataset.index)
            )
        )

        if args.save_data:
            dataset.to_pickle(args.pklname)

        if INTERRUPTED:
            break

        # increment batch number
        batch_num += 1

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
        plotter.standard("", False)
        plotter.show()
