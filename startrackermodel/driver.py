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
from os.path import isfile
from time import perf_counter

from classes import parameter as par
from classes.hardware import Hardware
from classes.software import Software
from classes.environment import Environment
from classes.estimation import Estimation
from classes.plotter import Plotter
from scripts.composer import Composer, SimType
from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger("driver")


def parse_arguments() -> argparse.Namespace:
    """
    Parse arguments from command line. If none are recevied, "
    "open dialog box for user to enter in information.

    Returns:
        arparse.Namespace: struct containing command line flags and arguments
    """

    # program info
    parser = argparse.ArgumentParser(
        prog="python -m driver",
        usage="python -m driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Estimate accuracy and precision of a star tracker via error propagation from hardware, software, and environmental sources."
                                                                
                                                            
                                                            
                                                            
                                                  +         
                                              .-#@@         
                                   @@+           @@ .::     
                                  :=-=           .          
                                                            
                         @@                                 
                 +@@@   :@@@                                
              #@@@@@@@: @@@@@            @@                 
           +@@@@@@@@@@@@@@@@@@*         : .-                
        #@@@@@@@@@@@@@@@@@@@@@@%                            
     -@@@@@@@@@@@@@@@@@@@@@@@@@@@                           
   @@@@@@@@@@@@@@@@@@@@@@@@                                 
    @@@@@@@@@@@@@@@@@@@@@@@@+                               
     @@@@@@@@@@@@@@@@@@@@@@@@                               
      @@@@@@@@@@@@@@@@@@@@%                                 
       #@@@@@@@@@@@@@@@#                                    
        -@@@@@@@@@@@*                                       
          @@@@@@@+                                          
           @@@+                                             
                                                            
                                                            
                                                            """,
        epilog="Source and bug tracker found at the following site:\n\t"
        "https://github.com/gagandeepthapar/StarTrackerModel"
        "",
    )

    """
    Tool Meta-Parameters
    """
    parser.add_argument(
        "-log",
        "--logger",
        metavar="",
        type=str,
        help="Set logging level: "
        "(D)ebug, (I)nfo, (W)arning, (E)rror, or (C)ritical. Default (D)ebug.",
        default="Debug",
    )

    """
    Simulation Parameters
    """
    parser.add_argument(
        "-n",
        "--numruns",
        metavar="",
        type=int,
        help="Number of runs per batch for Monte Carlo Analysis. Defaults to 1000.",
        default=1_000,
    )

    parser.add_argument(
        "-seed",
        "--randomseed",
        metavar="",
        type=int,
        help="Set random seed to re-run simulations. Defaults to unset.",
        default=None,
    )

    parser.add_argument(
        "-sim",
        "--simtype",
        metavar="",
        type=str,
        help="Set simulation type to (M)onte Carlo or (S)ensitivity. Default to Monte Carlo.",
        default="M",
    )

    """
    Simulation Object Parameters
    """
    parser.add_argument(
        "-hw",
        "--hardware",
        metavar="",
        type=str,
        help="Select hardware configuration in data/hardware.json. Default to Ideal.",
        default="IDEAL",
    )

    parser.add_argument(
        "-sw",
        "--software",
        metavar="",
        type=str,
        help="Select software configuration in data/software.json. Default to Ideal.",
        default="IDEAL",
    )

    parser.add_argument(
        "-env",
        "--environment",
        metavar="",
        type=str,
        help="Select environment configuration in data/environment.json. Default to Ideal.",
        default="IDEAL",
    )

    parser.add_argument(
        "-est",
        "--estimation",
        metavar="",
        type=str,
        help="Select estimation certainty in data/estimation.json. Default to Ideal.",
        default="IDEAL",
    )

    """
    Plotting Parameters
    """
    parser.add_argument(
        "--showplot",
        # metavar="",
        # type=bool,
        help="Show results of accuracy and precision in plots. Defaults to true.",
        action="store_true",
    )

    parser.add_argument(
        "--saveplot",
        # metavar="",
        # type=bool,
        help="Save all results of accuracy and precision in plots. "
        "Saves in media/ folder in root directory. Defaults to false.",
        action="store_true",
    )

    args = parser.parse_args()

    # get logging level from dict stored in CONSTANTS
    logger.setLevel(CONSTANTS.level_hash.get((args.logger).upper()))
    logger.debug("CMD Arguments: %s", args)

    return args


def create_hardware(hardware_flag: str, default: str = "IDEAL") -> Hardware:
    """
    Instantiate Hardware Class based on user-supplied CLI flag

    Inputs:
        hardware_flag (str) : CLI flag indicating which dict from hardware.json to use
        default (str)       : default dict (IDEAL) in case user-supplied does not exist

    Returns:
        Hardware            : Hardware class containing startracker hardware parameters
    """
    with open("data/hardware.json", encoding="utf-8") as hwfp:
        hwdict = json.load(hwfp)

    if hardware_flag.upper() not in hwdict.keys():
        logger.warning(
            "%s not found in data/hardware.json config file. "
            "Continuing with %s hardware",
            hardware_flag,
            default,
        )
        hardware_flag = default

    # attempt to retrieve JSON dict of user-supplied data
    hwconfig = hwdict.get(hardware_flag.upper())  # top level dict in JSON
    par_dict: Dict[str, par.Parameter] = {
        comp_name: par.NormalParameter.from_dict({comp_name: hwconfig.get(comp_name)})
        for comp_name in hwconfig
    }
    return Hardware(par_dict)


if __name__ == "__main__":
    # parse arguments
    cmd_arguments = parse_arguments()

    # set random seed
    logger.info("Setting Seed: %s", str(cmd_arguments.randomseed))
    np.random.seed(cmd_arguments.randomseed)

    # Set sim type
    match cmd_arguments.simtype.upper():
        case "S" | "SENSITVITY":
            sim_type = SimType.SENSITIVITY
        case "M" | "MONTECARLO":
            sim_type = SimType.MONTE_CARLO
        case _:
            sim_type = SimType.MONTE_CARLO
    logger.info("Setting Sim Type: %s", str(sim_type))

    # Instantiate hardware
    logger.info("Instantiating Hardware")
    sim_hw = create_hardware(cmd_arguments.hardware)
    logger.debug(sim_hw)

    # Instantiate software
    logger.info("Instantiating Software")
    sim_sw = create_hardware(cmd_arguments.hardware)
    # logger.debug(sim_hw)

    # Instantiate estimation
    logger.info("Instantiating Estimation")
    sim_est = create_hardware(cmd_arguments.hardware)
    # logger.debug(sim_hw)

    # Instantiate environment
    logger.info("Instantiating Environment")
    sim_env = create_hardware(cmd_arguments.hardware)
    # logger.debug(sim_hw)

    # Pass into composer
    hw = Hardware({"FOCAL_LENGTH": NormalParameter("FOCAL_LENGTH", "mm", 24, 1.5)})
    composer = Composer(hw, hw, sim_type)
    df = composer.span(1000)
