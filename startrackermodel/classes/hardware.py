"""
hardware

Class to contain all hardware-associated parameters for a given run.
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

from data import CONSTANTS
from classes.component import Component
from classes.parameter import NormalParameter, Parameter

import numpy as np
import pandas as pd

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Hardware(Component):
    """
    Hardware class
    """

    def __init__(self, hardware_dict: Dict[str, Parameter]):
        """
        Initialize Hardware class
        """

        self.focal_length = hardware_dict.get(
            "FOCAL_LENGTH", NormalParameter("FOCAL_LENGTH", "px", 24, 0)
        )
        self.focal_array = hardware_dict.get(
            "FOCAL_ARRAY_X", NormalParameter("FOCAL_ARRAY_X", "px", 1024, 0)
        )
        self.focal_array = hardware_dict.get(
            "FOCAL_ARRAY_Y", NormalParameter("FOCAL_ARRAY_Y", "px", 1024, 0)
        )
        self.f_arr_eps_x = hardware_dict.get(
            "FOCAL_ARRAY_EPS_X", NormalParameter("FOCAL_ARRAY_EPS_X", "px", 0, 0)
        )
        self.f_arr_eps_y = hardware_dict.get(
            "FOCAL_ARRAY_EPS_Y", NormalParameter("FOCAL_ARRAY_EPS_Y", "px", 0, 0)
        )

        self.f_arr_eps_z = hardware_dict.get(
            "FOCAL_ARRAY_EPS_Z", NormalParameter("FOCAL_ARRAY_EPS_Z", "px", 0, 0)
        )

        self.f_arr_phi_x = hardware_dict.get(
            "FOCAL_ARRAY_PHI_X", NormalParameter("FOCAL_ARRAY_PHI_X", "deg", 0, 0)
        )

        self.f_arr_theta_y = hardware_dict.get(
            "FOCAL_ARRAY_THETA_Y",
            NormalParameter("FOCAL_ARRAY_THETA_Y", "deg", 0, 0),
        )

        self.f_arr_psi_z = hardware_dict.get(
            "FOCAL_ARRAY_PSI_Z", NormalParameter("FOCAL_ARRAY_PSI_Z", "deg", 0, 0)
        )

        self.max_vis_mag = hardware_dict.get(
            "MAX_VIS_MAG", NormalParameter("MAX_VIS_MAG", "mv", 10000, 0)
        )

        self.object_list: List[Parameter] = [
            self.focal_length,
            self.focal_array,
            self.f_arr_eps_x,
            self.f_arr_eps_y,
            self.f_arr_eps_z,
            self.f_arr_phi_x,
            self.f_arr_theta_y,
            self.f_arr_psi_z,
            self.max_vis_mag,
        ]
        return

    def modulate(self, num: int) -> pl.DataFrame:
        """
        Modulate all objects in class simultaneously

        Inputs:
            num (int): number of values to generate for each Parameter

        Returns:
            pl.DataFrame: Polars DataFrame containing all generated data
        """
        print([param._name for param in self.object_list])
        return pl.DataFrame(
            {param._name: param.modulate(num) for param in self.object_list}
        )


if __name__ == "__main__":
    n = par.NormalParameter("A", "-", 0, 1)
    h = Hardware(n, n, n, n, n, n, n, n, n)
    print(h)
    df = h.modulate(10)
    print(h.object_list)
    print(df)
