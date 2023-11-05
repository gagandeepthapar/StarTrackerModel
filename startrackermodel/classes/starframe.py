"""
starframe

Class to contain all scene-specific information 
Includes:
    - visible stars 
    - image noise 
    - error stackup

startrackermodel
"""

import logging
import logging.config
from typing import Tuple

from numpy.random import f

from data import CONSTANTS
from classes.attitude import Attitude

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class StarFrame:
    def __init__(self):
        self.full_star_catalog: pd.DataFrame = StarFrame.prep_star_catalog()

    def filter_stars(
        self, boresight: np.ndarray, max_fov: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove all stars outside FOV from source catalog.
        Return trimmed catalog

        Args:
            boresight (np.ndarray): boresight of star tracker
            max_fov (float): full-angle fov of star tracker

        Returns:
            np.ndarray: array of unit vectors in ECI of relevant stars
            np.ndarray: array of magnitudes of relevant stars
        """
        catalog = self.full_star_catalog[
            self.full_star_catalog.apply(
                lambda row: (
                    np.arccos(np.dot(row.UVEC_ECI, boresight)) <= (max_fov / 2)
                ),
                axis=1,
            )
        ]
        return catalog.UVEC_ECI.to_numpy(), catalog.v_magnitude.to_numpy()

    @staticmethod
    def prep_star_catalog() -> pd.DataFrame:
        """
        Read in star catalog and generate uvec data in eci

        Returns:
            pd.DataFrame: dataframe of YBSC catalog with all useful information
        """
        star_catalog: pd.DataFrame = pd.read_pickle(CONSTANTS.YBSC_PKL)
        star_catalog = star_catalog.drop(
            [
                "spectral_type_a",
                "spectral_type_b",
                "ascension_proper_motion",
                "declination_proper_motion",
            ],
            axis=1,
        )
        star_catalog["UVEC_ECI"] = star_catalog.apply(
            lambda row: Attitude.ra_dec_to_uvec(row.right_ascension, row.declination),
            axis=1,
        )
        star_catalog.catalog_number = star_catalog.catalog_number.astype("int")
        return star_catalog
