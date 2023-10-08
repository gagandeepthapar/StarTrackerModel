"""
component

Parent class to contain all component-associated parameters for a given run.
Parent to:
    - Hardware
    - Software 
    - Environment
    - Estimation
Provides common variables, methods for children to inherit
Includes:
    - Focal Length (mm) + deviations
    - Focal Array width/height (px; assumes square, 0 deviations)
    - Translational errors of Focal Array (px)
    - Rotational errors of Focal Array (deg)
    - Maximum visible star magnitude (mv)

startrackermodel
"""

import logging
import logging.config
from typing import List, Dict
from abc import ABC

import pandas as pd

from data import CONSTANTS
from classes import parameter as par


logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Component(ABC):
    """
    Component class
    """

    def __init__(self, component_dict: Dict[str, par.Parameter]):
        """
        Instantiate Component class
        """
        self.object_list: List[par.Parameter] = list(component_dict.values())

    def modulate(self, num: int) -> pd.DataFrame:
        """
        Modulate all objects in class simultaneously

        Inputs:
            num (int): number of values to generate for each Parameter

        Returns:
            pd.DataFrame: DataFrame containing all generated data
        """
        return pd.DataFrame(
            {param.name: param.modulate(num) for param in self.object_list}
        )

    def span(self, num: int) -> pd.DataFrame:
        """
        For sensitivity analysis, selectively set parameters to ideal or stochastic values

        Inputs:
            num (int): total number of data points to process. Divides num by number of parameters.

        Returns:
            pd.DataFrame: DataFrame containing generated data in ascending order
        """

        param_num = num // len(self.object_list)
        data_frame = pd.DataFrame()

        for param in self.object_list:
            check_name = param.name
            param_df = pd.DataFrame(
                {
                    mod_param.name: (
                        mod_param.span(param_num)
                        if mod_param.name is check_name
                        else mod_param.ideal(param_num)
                    )
                    for mod_param in self.object_list
                }
            )

            data_frame = pd.concat([data_frame, param_df], axis=0)

        return data_frame
