"""
attitude

Class to contain all attitude-associated parameters for a given run.
Includes:
    - Right Ascension
    - Declination
    - Roll Angle
    - Associated Unit Vector Boresight (uvec)
    - Associated Quaternion (qTrue)
    - Associated rotation matrix (R_STAR)

startrackermodel
"""

import logging
import logging.config
from typing import List, Dict
import numpy as np
import pandas as pd

from data import CONSTANTS
from classes import component as comp
from classes import parameter as par


logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Attitude(comp.Component):
    """
    Attitude class
    """

    def __init__(self):
        self.right_asc = par.UniformParameter("RIGHT_ASCENSION", "rad", 0, 2 * np.pi)
        self.dec = par.UniformParameter("DECLINATION", "rad", -np.pi / 2, np.pi / 2)
        self.roll = par.UniformParameter("ROLL", "rad", 0, 2 * np.pi)

        self.object_list: List[par.Parameter] = [self.right_asc, self.dec, self.roll]

    def modulate(self, num: int) -> pd.DataFrame:
        """
        Calculate unit vector of boresight and true quaternion from RA, DEC

        Inputs:
            num (int)   : number of fields to generate

        Returns:
            pl.Dataframe: DataFrame of RA, DEC, UVEC, QTRUE num number of rows
        """
        # get random pointing axis
        attitude_data = super().modulate(num)

        # complete attitude representation
        full_attitude_rep = Attitude.complete_attitude_repr(attitude_data)

        return full_attitude_rep

    def span(self, num: int) -> pd.DataFrame:
        """
        Span across all attitudes with total n rows

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF with ascending data in RA then DEC
        """
        # gen randomized data
        attitude_data = super().span(num)

        # complte attitude repr
        full_attitude_rep = Attitude.complete_attitude_repr(attitude_data)

        return full_attitude_rep

    @staticmethod
    def ra_dec_to_uvec(right_asc: float, dec: float) -> np.ndarray:
        """
        Convert right ascension, declination into unit vector

        Inputs:
            right_asc (float)  : Right Ascension [rad]
            dec (float) : Declination [rad]

        Returns:
            np.ndarray  : unit vector in same frame as inputs
        """
        return np.array(
            [
                np.cos(right_asc) * np.cos(dec),
                np.sin(right_asc) * np.cos(dec),
                np.sin(dec),
            ]
        )

    @staticmethod
    def complete_attitude_repr(ra_dec_roll_data: pd.DataFrame) -> pd.DataFrame:
        """
        Attitude Class only modulates RA, DEC, ROLL. Use this to compute
        UVEC, R_STAR, and Q_TRUE

        Inputs:
            ra_dec_roll_data (pd.DataFrame): DataFrame of randomized data

        Returns:
            pd.DataFrame: updated DF with UVEC, R_STAR, Q_TRUE
        """
        # calc uvec
        ra_dec_roll_data["UVEC_ECI"] = ra_dec_roll_data.apply(
            lambda row: Attitude.ra_dec_to_uvec(row.RIGHT_ASCENSION, row.DECLINATION),
            axis=1,
        )

        # calc rotation matrix
        ra_dec_roll_data["R_STAR"] = ra_dec_roll_data.apply(
            lambda row: Attitude.ra_dec_to_rotm(
                row.RIGHT_ASCENSION, row.DECLINATION, row.ROLL
            ),
            axis=1,
        )

        # calc q_true
        ra_dec_roll_data["Q_TRUE"] = ra_dec_roll_data["R_STAR"].apply(
            Attitude.rotm_to_quat
        )

        return ra_dec_roll_data

    @staticmethod
    def ra_dec_to_rotm(right_asc: float, dec: float, roll: float) -> np.ndarray:
        """
        Converts full attitude in 3-angle system to unique rotation matrix

        Args:
            right_asc (float): Right Ascension [rad]
            dec (float): Declination Angle [rad]
            roll (float): Roll angle [rad]

        Returns:
            np.ndarray: 3x3 Rotation matrix to rotate from ECI into body-frame (assuming Z-axis is along boresight)
        """

        return (
            Attitude.rotm_z(roll)
            @ Attitude.rotm_y(dec - np.pi / 2)
            @ Attitude.rotm_z(-right_asc)
        )

    @staticmethod
    def rotm_x(phi: float) -> np.ndarray:
        """
        Create principal axis rotm about x axis

        Args:
            phi (float): angle to rotate

        Returns:
            np.ndarray: 3x3 rotm
        """
        return np.array(
            [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
        )

    @staticmethod
    def rotm_y(phi: float) -> np.ndarray:
        """
        Create principal axis rotm about x axis

        Args:
            phi (float): angle to rotate

        Returns:
            np.ndarray: 3x3 rotm
        """
        return np.array(
            [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]
        )

    @staticmethod
    def rotm_z(phi: float) -> np.ndarray:
        """
        Create principal axis rotm about x axis

        Args:
            phi (float): angle to rotate

        Returns:
            np.ndarray: 3x3 rotm
        """
        return np.array(
            [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
        )

    @staticmethod
    def rotm_to_quat(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix into quaternion

        Inputs:
            R (np.ndarray): Rotation matrix from A to B

        Returns:
            q (np.ndarray): quaternion from A to B
        """
        n = np.sqrt(R.trace() + 1) / 2
        e_1 = (R[1, 2] - R[2, 1]) / (4 * n)
        e_2 = (R[2, 0] - R[0, 2]) / (4 * n)
        e_3 = (R[0, 1] - R[1, 0]) / (4 * n)
        return np.array([e_1, e_2, e_3, n])


if __name__ == "__main__":
    at = Attitude()
    df = at.modulate(10)
    logger.info(df.columns)
