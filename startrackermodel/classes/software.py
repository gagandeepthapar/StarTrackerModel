"""
software

Model the software accuracy of the star tracker algorithm.
Includes:
    - Centroiding Error
    - Computation Time (forces IMUs to propagate attitude forward)

startrackermodel
"""
from time import perf_counter
import logging
import logging.config

from typing import Dict

from data import CONSTANTS
from classes.component import Component
from classes.parameter import Parameter, UniformParameter, NormalParameter

import numpy as np
import pandas as pd

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Software(Component):
    # basic error based on magnitude (px)
    base_acc = lambda M: np.random.normal(
        0, (np.exp(0.499421 * M - 5.16725) + 0.00290995)
    )

    def __init__(self, software_cfg: Dict[str, Parameter]):
        """
        Initialize Software class
        """
        self.error_multiplier = software_cfg.get(
            "ERROR_MULTIPLIER", NormalParameter("ERROR_MULTIPLIER", "-", 1, 0.1)
        )
        # self.error_direction = UniformParameter("ERROR_DIRECTION", "rad", 0, 2 * np.pi)

        self.object_list = [self.error_multiplier]
        return

    def modulate(self, num: int) -> pd.DataFrame:
        """
        Update modulate base class with derived parameters

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF
        """
        sw_df = super().modulate(num)
        return sw_df

    def span(self, num: int) -> pd.DataFrame:
        """
        Update modulate base class with derived parameters

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF
        """
        sw_df = super().span(num)
        return sw_df

    def __repr__(self) -> str:
        """
        Return representation of class
        """
        return f"Software Class:\n{super().__repr__()}"


if __name__ == "__main__":
    base_acc = lambda M: np.exp(0.499421 * M - 5.16725) + 0.00290995

    r = np.linspace(0, 7, 1_000)
    s = perf_counter()
    rp = base_acc(r)
    e = perf_counter()
    print(e - s)
    print(rp.shape)
