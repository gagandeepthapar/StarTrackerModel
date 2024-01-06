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
from os.path import isfile, join
from datetime import datetime

from classes import parameter as par
from classes.component import Component
from classes.hardware import Hardware
from classes.software import Software
from classes.sensor import Sensor
from classes.environment import Environment
from classes.enums import SimType, ComponentType
from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger("driver")


class UserArguments:
    def __init__(self):
        self.__cmd_arguments = self.parse_arguments()
        self.default_cfgs = {
            ComponentType.HARDWARE: "data/hardware.json",
            ComponentType.SOFTWARE: "data/software.json",
            ComponentType.SENSOR: "data/sensor.json",
            ComponentType.ENVIRONMENT: "data/environment.json",
        }

        """ 
        Process Arguments
        """
        self.num_runs = self.__cmd_arguments.numruns
        self.max_runs = self.__cmd_arguments.maxruns
        self.eps = self.__cmd_arguments.eps
        self.show_plot = self.__cmd_arguments.showplot
        self.save_plot = self.__cmd_arguments.saveplot
        self.save_data = self.__cmd_arguments.savedata

        self.param_select = self.__cmd_arguments.select

        now = datetime.now()
        self.pklname = join(
            CONSTANTS.SAVEDATA, now.strftime("%Y_%m_%d_%H_%M_%S") + ".pkl"
        )
        if self.save_data:
            logger.info(f"Saving data to {self.pklname}")

        # set random seed
        logger.info("Setting Seed: %s", str(self.__cmd_arguments.randomseed))
        np.random.seed(self.__cmd_arguments.randomseed)

        self.sim_type = self.set_sim_type(self.__cmd_arguments.simtype)

        # Instantiate hardware
        logger.info("Instantiating Hardware")
        self.sim_hw = self.create_component(
            self.__cmd_arguments.hardware, ComponentType.HARDWARE, "data/hardware.json"
        )
        logger.debug(self.sim_hw)

        # Instantiate software
        logger.info("Instantiating Software")
        self.sim_sw = self.create_component(
            self.__cmd_arguments.software, ComponentType.SOFTWARE, "data/software.json"
        )
        logger.debug(self.sim_sw)

        # Instantiate sensor
        logger.info("Instantiating Sensor")
        self.sensor = self.create_component(
            self.__cmd_arguments.sensor, ComponentType.SENSOR, "data/sensor.json"
        )
        logger.debug(self.sensor)

        # Instantiate environment
        logger.info("Instantiating Environment")
        self.env = self.create_component(
            self.__cmd_arguments.environment,
            ComponentType.ENVIRONMENT,
            "data/environment.json",
        )
        logger.debug(self.env)

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
            default="Info",
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

        parser.add_argument(
            "--select",
            metavar="",
            type=str,
            nargs="+",
            help="Select which parameters to vary for the Sensitivity analysis. Defaults to all.",
            default=None,
        )

        parser.add_argument(
            "--savedata",
            # metavar="",
            # type=bool,
            help="Save all results of accuracy and precision as pkl file. "
            "Saves in data/ folder in root directory. Defaults to false.",
            action="store_true",
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
            "-sens",
            "--sensor",
            metavar="",
            type=str,
            help="Select sensor configuration in data/sensor.json. Default to Ideal.",
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

    def create_component(
        self,
        component_flag: str,
        component_type: ComponentType,
        cfg_file: str = None,
        default: str = "IDEAL",
    ) -> Component:
        """
        Instantiate Component Class based on user-supplied CLI flag

        Inputs:
            component_flag (str) : CLI flag indicating which dict from cfg_file to use
            cfg_file       (str) : file location of cfg file if supplied
            component_type (ComponentType) : indication of which component to create
            default (str)       : default dict (IDEAL) in case user-supplied does not exist

        Returns:
            Component            : Component class containing startracker parameters
        """

        if cfg_file is None:
            cfg_file = self.default_cfgs.get(component_type)

        if not isfile(cfg_file):
            logger.warning(
                "%s file not found. " "Continuing with default config file", cfg_file
            )
            cfg_file = self.default_cfgs.get(component_type)

        with open(cfg_file, encoding="utf-8") as hwfp:
            comp_dict = json.load(hwfp)

        if component_flag.upper() not in comp_dict.keys():
            logger.warning(
                "%s not found in %s config file. " "Continuing with %s hardware",
                component_flag,
                cfg_file,
                default,
            )
            component_flag = default

        # attempt to retrieve JSON dict of user-supplied data
        comp_performance_level = comp_dict.get(component_flag.upper())
        par_dict: Dict[str, par.Parameter] = {}

        # create uniform and normal parameters; default to normal
        for param_var_name in comp_performance_level:
            if comp_performance_level.get(param_var_name).get("TYPE", "N").upper() in [
                "N",
                "NORMAL",
            ]:
                par_dict[param_var_name] = par.NormalParameter.from_dict(
                    {param_var_name: comp_performance_level.get(param_var_name)}
                )
            else:
                par_dict[param_var_name] = par.UniformParameter.from_dict(
                    {param_var_name: comp_performance_level.get(param_var_name)}
                )

        match component_type:
            case ComponentType.HARDWARE:
                return Hardware(par_dict)
            case ComponentType.SOFTWARE:
                return Software(par_dict)
            case ComponentType.SENSOR:
                return Sensor(par_dict)
            case ComponentType.ENVIRONMENT:
                return Environment(par_dict)
