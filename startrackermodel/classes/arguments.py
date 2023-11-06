"""
arguments

Class to unify user-arguments from different spaces into singular interface

startrackermodel
"""
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


class UserArguments:
    def __init__(self):
        self.__cmd_arguments = self.parse_arguments()

        """ 
        Process Arguments
        """
        self.num_runs = self.__cmd_arguments.numruns
        self.max_runs = self.__cmd_arguments.maxruns
        self.eps = self.__cmd_arguments.eps
        self.show_plot = self.__cmd_arguments.showplot
        self.save_plot = self.__cmd_arguments.saveplot

        # set random seed
        logger.info("Setting Seed: %s", str(self.__cmd_arguments.randomseed))
        np.random.seed(self.__cmd_arguments.randomseed)

        self.sim_type = self.set_sim_type(self.__cmd_arguments.simtype)

        # Instantiate hardware
        logger.info("Instantiating Hardware")
        self.sim_hw = self.create_hardware(self.__cmd_arguments.hardware)
        logger.debug(self.sim_hw)

        # Instantiate software
        logger.info("Instantiating Software")
        self.sim_sw = self.create_software(self.__cmd_arguments.software)
        logger.debug(self.sim_sw)

        # Instantiate estimation
        logger.info("Instantiating Estimation")
        # sim_est = create_hardware(__cmd_arguments.hardware)
        # logger.debug(sim_hw)

        # Instantiate environment
        logger.info("Instantiating Environment")
        # sim_env = create_hardware(__cmd_arguments.hardware)
        # logger.debug(sim_hw)

        return

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse arguments from command line.

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
            help="Number of runs per batch for Monte Carlo Analysis. Defaults to 1,000.",
            default=1_000,
        )

        parser.add_argument(
            "-max",
            "--maxruns",
            metavar="",
            type=int,
            help="Maximum number of runs to compute for Monte Carlo Analysis. Defaults to inf.",
            default=np.inf,
        )

        parser.add_argument(
            "-eps",
            metavar="",
            type=float,
            help="Tolerance for stopping criterion. Smaller = more runs. Defaults to 1e-3.",
            default=1e-3,
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
        logger.setLevel(CONSTANTS.level_hash.get((args.logger).upper()))  # type: ignore
        logger.debug("CMD Arguments: %s", args)

        return args

    def set_sim_type(self, sim_arg: str) -> SimType:
        # Set sim type
        match sim_arg.upper():
            case "S" | "SENSITVITY":
                sim_type = SimType.SENSITIVITY
            case "M" | "MONTECARLO":
                sim_type = SimType.MONTE_CARLO
            case _:
                sim_type = SimType.MONTE_CARLO
        logger.info("Setting Sim Type: %s", str(sim_type))
        return sim_type

    def create_hardware(
        self,
        hardware_flag: str,
        default: str = "IDEAL",
        cfg_file: str = "data/hardware.json",
    ) -> Hardware:
        """
        Instantiate Hardware Class based on user-supplied CLI flag

        Inputs:
            hardware_flag (str) : CLI flag indicating which dict from hardware.json to use
            default (str)       : default dict (IDEAL) in case user-supplied does not exist

        Returns:
            Hardware            : Hardware class containing startracker hardware parameters
        """
        if not isfile(cfg_file):
            logger.warning(
                "%s file not found. " "Continuing with data/hardware.json", cfg_file
            )
            cfg_file = "data/hardware.json"

        with open(cfg_file, encoding="utf-8") as hwfp:
            hwdict = json.load(hwfp)

        if hardware_flag.upper() not in hwdict.keys():
            logger.warning(
                "%s not found in %s config file. " "Continuing with %s hardware",
                hardware_flag,
                cfg_file,
                default,
            )
            hardware_flag = default

        # attempt to retrieve JSON dict of user-supplied data
        hwconfig = hwdict.get(hardware_flag.upper())  # top level dict in JSON
        par_dict: Dict[str, par.Parameter] = {
            comp_name: par.NormalParameter.from_dict(
                {comp_name: hwconfig.get(comp_name)}
            )
            for comp_name in hwconfig
        }
        return Hardware(par_dict)

    def create_software(
        self,
        software_flag: str,
        default: str = "IDEAL",
        cfg_file: str = "data/software.json",
    ) -> Software:
        """
        Instantiate Hardware Class based on user-supplied CLI flag

        Inputs:
            hardware_flag (str) : CLI flag indicating which dict from hardware.json to use
            default (str)       : default dict (IDEAL) in case user-supplied does not exist
            json (str)          : file location of json file to read. Default to data/software.json

        Returns:
            Hardware            : Hardware class containing startracker hardware parameters
        """
        if not isfile(cfg_file):
            logger.warning(
                "%s file not found. " "Continuing with data/software.json", cfg_file
            )
            cfg_file = "data/software.json"

        with open(cfg_file, encoding="utf-8") as swfp:
            swdict = json.load(swfp)

        if software_flag.upper() not in swdict.keys():
            logger.warning(
                "%s not found in %s config file. " "Continuing with %s software",
                software_flag,
                cfg_file,
                default,
            )
            software_flag = default

        # attempt to retrieve JSON dict of user-supplied data
        swconfig = swdict.get(software_flag.upper())  # top level dict in JSON
        par_dict: Dict[str, par.Parameter] = {
            comp_name: par.NormalParameter.from_dict(
                {comp_name: swconfig.get(comp_name)}
            )
            for comp_name in swconfig
        }
        return Software(par_dict)
