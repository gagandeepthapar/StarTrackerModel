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
import argparse
import json
from time import perf_counter, perf_counter_ns
import numpy as np

from classes import parameter as par
from classes.hardware import Hardware
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
        metavar="",
        type=bool,
        help="Show results of accuracy and precision in plots. Defaults to true.",
        default=True,
    )

    parser.add_argument(
        "--saveplot",
        metavar="",
        type=bool,
        help="Save all results of accuracy and precision in plots. "
        "Saves in media/ folder in root directory. Defaults to false.",
        default=False,
    )

    args = parser.parse_args()

    # get logging level from dict stored in CONSTANTS
    logger.setLevel(CONSTANTS.level_hash.get((args.logger).upper()))
    logger.debug("CMD Arguments: %s", args)

    return args


def create_hardware(hardware_flag: str, default: str = "IDEAL") -> Hardware:
    with open("data/hardware.json") as hwfp:
        hwdict = json.load(hwfp)

    if hardware_flag.upper() not in hwdict.keys():
        logger.warning(
            "%s not found in data/hardware.json config file."
            "Continuing with %s hardware",
            hardware_flag,
            default,
        )

    # attempt to retrieve JSON dict of user-supplied data
    hwconfig = hwdict.get(hardware_flag.upper(), default)
    par_dict = {
        k: par.NormalParameter.from_dict({k: hwconfig.get(k)}) for k in hwconfig
    }
    return Hardware(par_dict)


if __name__ == "__main__":
    # parse arguments
    cmd_arguments = parse_arguments()

    # set random seed
    np.random.seed(cmd_arguments.randomseed)

    # Instantiate hardware
    sim_hw = create_hardware(cmd_arguments.hardware)

    # Instantiate software

    # Instantiate estimation

    # Instantiate environment

    # Pass into composer
