"""
component

Parent class to contain all component-associated parameters for a given run.
Parent to:
    - Hardware
    - Software 
    - Environment
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

    def span(self, num: int, param_select: List[str] = None) -> pd.DataFrame:
        """
        For sensitivity analysis, selectively set parameters to ideal or stochastic values

        Inputs:
            num (int): total number of data points to process. Divides num by number of parameters.

        Returns:
            pd.DataFrame: DataFrame containing generated data in ascending order
        """

        # param_num = num // len(self.object_list)
        data_frame = pd.DataFrame()

        if param_select is None:
            param_select = [param.name for param in self.object_list]

        for param in self.object_list:
            check_name = param.name
            if check_name not in param_select:
                continue
            param_df = pd.DataFrame(
                {
                    mod_param.name: (
                        mod_param.span(num)
                        if mod_param.name is check_name
                        else mod_param.ideal(num)
                    )
                    for mod_param in self.object_list
                }
            )

            data_frame = pd.concat([data_frame, param_df], axis=0)

        return data_frame

    def __repr__(self) -> str:
        """
        Return representation of class
        """
        comp_repr = ""
        for comp_obj in self.object_list:
            comp_repr += "\t" + str(comp_obj) + "\n"
        return f"{comp_repr}"
