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
        full_hw_df = Hardware.precompute_metrics(hw_df)
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
        full_hw_df = Hardware.precompute_metrics(hw_df)
        return full_hw_df

    @staticmethod
    def precompute_metrics(hardware_data: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute metrics for hardware errors

        Inputs:
            hardware_data (pd.DataFrame): dataframe of hardware values

        Returns:
            pd.DataFrame: DF with fov, r_gamma_pi matrix
        """

        hardware_data["MAX_FOV"] = Hardware.compute_fov(
            hardware_data[["FOCAL_ARRAY_X", "FOCAL_ARRAY_Y"]].apply(
                np.linalg.norm, axis=1
            ),  # type: ignore
            hardware_data.FOCAL_LENGTH,  # type: ignore
        )

        hardware_data["R_GAMMA_PI"] = hardware_data[
            ["FOCAL_ARRAY_THETA_Z", "FOCAL_ARRAY_THETA_Y", "FOCAL_ARRAY_THETA_X"]
        ].apply(
            lambda row: Attitude.rotm_z(CONSTANTS.DEG2RAD * row.FOCAL_ARRAY_THETA_Z)
            @ Attitude.rotm_y(CONSTANTS.DEG2RAD * row.FOCAL_ARRAY_THETA_Y)
            @ Attitude.rotm_x(CONSTANTS.DEG2RAD * row.FOCAL_ARRAY_THETA_X),
            axis=1,
        )

        hardware_data["R_F_GAMMA_GAMMA"] = hardware_data.apply(
            lambda row: Hardware.get_r_f_gamma_gamma(
                row.FOCAL_LENGTH,
                np.array(
                    [
                        row.FOCAL_ARRAY_DELTA_X,
                        row.FOCAL_ARRAY_DELTA_Y,
                        row.FOCAL_ARRAY_DELTA_Z,
                    ]
                ),
                row.R_GAMMA_PI,
            ),
            axis=1,
        )

        return hardware_data

    @staticmethod
    def get_r_f_gamma_gamma(
        flen: float, delv: np.ndarray, r_gamma_pi: np.ndarray
    ) -> np.ndarray:
        """
        compute vector to focal point from gamma plane

        Args:
            flen (float): focal length in pi system
            delv (np.ndarray): translation to gamma from pi in pi system
            r_gamma_pi (np.ndarray): rotation to gamma from pi

        Returns:
            np.ndarray: vector to focal point from gamma plane
        """
        r_f_pi_pi = np.array([0, 0, flen])
        r_f_gamma_pi = r_f_pi_pi - delv
        mag = np.linalg.norm(r_f_gamma_pi)
        dir_gam = r_gamma_pi @ Attitude.unit(r_f_gamma_pi)
        return mag * dir_gam

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
        return 2 * np.arctan(0.5 * focal_array / focal_length)

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
