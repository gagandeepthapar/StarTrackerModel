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

    def modulate(self, num: int) -> pd.DataFrame:
        """
        Update modulate base class with derived parameters

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF with FOV column
        """
        hw_df = super().modulate(num)
        full_hw_df = Hardware.complete_fov_cols(hw_df)
        return full_hw_df

    def span(self, num: int) -> pd.DataFrame:
        """
        Update modulate base class with derived parameters

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF with FOV column
        """
        hw_df = super().modulate(num)
        full_hw_df = Hardware.complete_fov_cols(hw_df)
        return full_hw_df

    @staticmethod
    def complete_fov_cols(hardware_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute FOV based on focal array and focal length

        Inputs:
            hardware_data (pd.DataFrame): dataframe of hardware values

        Returns:
            pd.DataFrame: DF with MAX_FOV, FOV_X, FOV_Y columns
        """

        hardware_data["MAX_FOV"] = Hardware.compute_fov(
            hardware_data[["FOCAL_ARRAY_X", "FOCAL_ARRAY_Y"]].apply(
                np.linalg.norm, axis=1
            ),
            hardware_data.FOCAL_LENGTH,
        )
        hardware_data["FOV_X"] = Hardware.compute_fov(
            hardware_data.FOCAL_ARRAY_X, hardware_data.FOCAL_LENGTH
        )
        hardware_data["FOV_Y"] = Hardware.compute_fov(
            hardware_data.FOCAL_ARRAY_Y, hardware_data.FOCAL_LENGTH
        )
        return hardware_data

    @staticmethod
    def compute_fov(focal_array: float, focal_length: float) -> float:
        """
        Compute FOV of hardware based on focal array, focal length

        Inputs:
            focal_array (float): number of pixels on edge of array
            focal_length (float): focal_length in pixels

        Returns:
            float: full FOV of hardware
        """
        return 2 * np.arctan2(0.5 * focal_array, focal_length)

    def __repr__(self) -> str:
        """
        Return representation of class
        """
        comp_repr = ""
        for comp_obj in self.object_list:
            comp_repr += "\t" + str(comp_obj) + "\n"
        return f"Hardware Class:\n{comp_repr}"


if __name__ == "__main__":
    n = NormalParameter("A", "-", 0, 1)
