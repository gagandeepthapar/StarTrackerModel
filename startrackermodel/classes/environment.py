"""
environment

Class to contain all environment-associated meters for a given run.
Includes:
    - Local temperature 
    - Local radiation 
    - Local velocity (circular orbits) 

startrackermodel
"""

import logging
import logging.config
from typing import List, Dict, Tuple

from data import CONSTANTS
from classes.component import Component
from classes.parameter import NormalParameter, Parameter

import numpy as np
import pandas as pd

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Environment(Component):
    """
    Environment class
    """

    def __init__(self, environment_cfg: Dict[str, Parameter]):
        """
        Initialize Environment class
        """
        self.emissivity = environment_cfg.get(
            "EMISSIVITY", NormalParameter("EMISSIVITY", "-", 0.25, 0)
        )

        self.object_list: List[Parameter] = [
            self.emissivity,
        ]

    def modulate(self, num: int) -> pd.DataFrame:
        """
        Update modulate base class with derived parameters

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF with speed, rad, column
        """
        hw_df = super().modulate(num)
        full_hw_df = Environment.precompute_metrics(hw_df)
        return full_hw_df

    def span(self, num: int) -> pd.DataFrame:
        """
        Update modulate base class with derived parameters

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF with FOV column
        """
        hw_df = super().span(num)
        full_hw_df = Environment.precompute_metrics(hw_df)
        return full_hw_df

    @staticmethod
    def temp_trends(emissivity: float) -> Tuple[float, float]:
        t_min_trend = -61.532 * emissivity + -1.54993
        t_max_trend = -62.7539 * emissivity + 41.3659

        return (t_min_trend, t_max_trend)

    @staticmethod
    def precompute_metrics(environment_df: pd.DataFrame) -> pd.DataFrame:
        # compute local temperature
        environment_df["TEMPERATURE"] = np.array(
            [
                np.random.uniform(*Environment.temp_trends(emis)) + CONSTANTS.C_TO_K
                for emis in environment_df.EMISSIVITY
            ]
        )

        return environment_df


if __name__ == "__main__":
    e = Environment({})
    logger.debug(e.modulate(5))
