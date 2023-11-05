"""
hardware

Class to contain all hardware-associated meters for a given run.
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
from classes.attitude import Attitude

import numpy as np
import pandas as pd

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Hardware(Component):
    """
    Hardware class
    """

    def __init__(self, hardware_cfg: Dict[str, Parameter]):
        """
        Initialize Hardware class
        """
        self.focal_length = hardware_cfg.get(
            "FOCAL_LENGTH", NormalParameter("FOCAL_LENGTH", "px", 3500, 0)
        )
        self.focal_array_x = hardware_cfg.get(
            "FOCAL_ARRAY_X", NormalParameter("FOCAL_ARRAY_X", "px", 1024, 0)
        )
        self.focal_array_y = hardware_cfg.get(
            "FOCAL_ARRAY_Y", NormalParameter("FOCAL_ARRAY_Y", "px", 1024, 0)
        )
        self.f_arr_eps_x = hardware_cfg.get(
            "FOCAL_ARRAY_DELTA_X", NormalParameter("FOCAL_ARRAY_DELTA_X", "px", 0, 0)
        )
        self.f_arr_eps_y = hardware_cfg.get(
            "FOCAL_ARRAY_DELTA_Y", NormalParameter("FOCAL_ARRAY_DELTA_Y", "px", 0, 0)
        )

        self.f_arr_eps_z = hardware_cfg.get(
            "FOCAL_ARRAY_DELTA_Z", NormalParameter("FOCAL_ARRAY_DELTA_Z", "px", 0, 0)
        )

        self.f_arr_theta_x = hardware_cfg.get(
            "FOCAL_ARRAY_THETA_X", NormalParameter("FOCAL_ARRAY_THETA_X", "rad", 0, 0)
        )
        self.f_arr_theta_x._stddev *= CONSTANTS.DEG2RAD  # type: ignore
        self.f_arr_theta_x._units = "rad"

        self.f_arr_theta_y = hardware_cfg.get(
            "FOCAL_ARRAY_THETA_Y",
            NormalParameter("FOCAL_ARRAY_THETA_Y", "rad", 0, 0),
        )
        self.f_arr_theta_y._stddev *= CONSTANTS.DEG2RAD  # type: ignore
        self.f_arr_theta_y._units = "rad"

        self.f_arr_theta_z = hardware_cfg.get(
            "FOCAL_ARRAY_THETA_Z", NormalParameter("FOCAL_ARRAY_THETA_Z", "rad", 0, 0)
        )
        self.f_arr_theta_z._stddev *= CONSTANTS.DEG2RAD  # type:ignore
        self.f_arr_theta_z._units = "rad"

        self.max_vis_mag = hardware_dict.get(
            "MAX_VIS_MAG", NormalParameter("MAX_VIS_MAG", "mv", 10000, 0)
        )
        self.hw_data: pd.DataFrame = pd.DataFrame()

        self.object_list: List[Parameter] = [
            self.focal_length,
            self.focal_array_x,
            self.focal_array_y,
            self.f_arr_eps_x,
            self.f_arr_eps_y,
            self.f_arr_eps_z,
            self.f_arr_theta_x,
            self.f_arr_theta_y,
            self.f_arr_theta_z,
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
