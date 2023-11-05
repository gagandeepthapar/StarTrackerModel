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
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import CONSTANTS
from classes.hardware import Hardware
from classes.software import Software
from classes.environment import Environment
from classes.estimation import Estimation
from classes.attitude import Attitude
from classes.parameter import Parameter
from classes.component import Component
from classes.starframe import StarFrame
from classes.enums import SimType

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Composer:
    """
    Converts user-supplied arguments into actionable DataFrames
    """

    def __init__(
        self,
        hardware: Hardware,
        software: Hardware,
        # environment: Environment,
        # estimation: Estimation,
        sim_type: SimType,
    ):
        self.hardware = hardware
        self.software = software
        # self.environment = environment
        # self.estimation = estimation
        self.attitude = Attitude()
        self.sim_type = sim_type

        self.components: List[Component] = [
            self.attitude,
            self.hardware,
            self.software,
            # self.environment,
            # self.estimation,
        ]

        self.starframe = StarFrame()

    def generate_data(self, num: int) -> pd.DataFrame:
        """
        Generate sim-specific data and store in class

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: Dataframe of data
        """

        data_df = pd.DataFrame()
        match self.sim_type:
            case SimType.MONTE_CARLO:
                data_df = self.modulate(num)

            case SimType.SENSITIVITY:
                data_df = self.span(num)

        self.frontend_data = data_df
        return data_df

    def modulate(self, num: int) -> pd.DataFrame:
        """
        Generate all data based on user-supplied information

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: Df of all columns randomized according to distribution
        """
        data_df = pd.concat([comp.modulate(num) for comp in self.components], axis=1)
        return data_df

    def span(self, num: int) -> pd.DataFrame:
        """
        Generate Sensitivity Analysis inputs

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF of Sensitivity analysis data
        """
        # extract all relevant parameters from child components
        total_objects: List[Parameter] = []
        for comp in self.components[1:]:
            total_objects.extend(comp.object_list)

        # span each parameter based on total number of parameters
        param_num = num // len(total_objects)
        data_frame = pd.DataFrame()
        logger.warning(param_num)
        for param in total_objects:
            check_name = param.name
            param_df = pd.DataFrame(
                {
                    mod_param.name: (
                        mod_param.span(param_num)
                        if mod_param.name is check_name
                        else mod_param.ideal(param_num)
                    )
                    for mod_param in total_objects
                }
            )

            data_frame = pd.concat([data_frame, param_df], axis=0)

        data_frame.reset_index(inplace=True)
        data_frame = pd.concat(
            [self.attitude.modulate(len(data_frame.index)), data_frame],
            axis=1,
        )
        logger.critical(data_frame)
        return data_frame


if __name__ == "__main__":
    comp = Composer(Hardware({}), Hardware({}), SimType.MONTE_CARLO)
    logger.debug(comp.modulate(10).columns)
