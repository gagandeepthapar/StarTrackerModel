"""
parameter

Parameter classes define the inputs with different distributions.
Generate n values following the intended distribution and perform vectorized math.

startrackermodel
"""

import logging
import logging.config

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Parameter(ABC):
    """
    Abstract class to provide function stubs to other functions
    """

    def __init__(self, name: str, units: str):
        self.name = name
        self._units = units
        self._sym = "X"

    def __repr__(self) -> str:
        return f"{self.name} [{self._units}]: {self._sym}"

    @abstractmethod
    def ideal(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] size array of values of central value between low and high

        Inputs:
            num (int): number of values to generate

        Returns:
            np.ndarray: array of "ideal" values
        """

    @abstractmethod
    def modulate(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] size array of values following distribution set in init

        Inputs:
            num(int): number of values to generate

        Returns:
            np.ndarray: array of values following distribution

        """

    @abstractmethod
    def span(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] array of values spanning [-5 sigma, 5sigma] of parameter

        Inputs:
            num (int)   : number of values in array

        Returns:
            np.ndarray  : array of values spanning the range
        """


class NormalParameter(Parameter):
    """
    Parameter to generate values following normal distribution
    """

    def __init__(self, name: str, units: str, mean: float, stddev: float):
        """
        Initialize Normal Parameter

        Inputs:
            name (str): Name of parameter
            units (str): units of parameter
            mean (float): mean of distribution
            stddev (float): stddev of distribution
        """
        super().__init__(name, units)
        self._mean = mean
        self._stddev = stddev
        self._sym = "N"

    def __repr__(self) -> str:
        """
        Representation of Normal Parameter
        """
        return (
            "[Normal Parameter] "
            + super().__repr__()
            + f"({self._mean}, {self._stddev}**2)"
        )

    def ideal(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] size array of values of central value between low and high

        Inputs:
            num (int): number of values to generate

        Returns:
            np.ndarray: array of "ideal" values
        """
        return np.ones(num) * self._mean

    def modulate(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] size array of values following distribution set in init

        Inputs:
            num(int): number of values to generate

        Returns:
            np.ndarray: array of values following distribution

        """
        return np.random.normal(self._mean, self._stddev, num)

    def span(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] array of values spanning [-5 sigma, 5sigma] of parameter

        Inputs:
            num (int)   : number of values in array

        Returns:
            np.ndarray  : array of values spanning the range
        """
        return np.linspace(
            self._mean - 5 * self._stddev, self._mean + 5 * self._stddev, num
        )

    @staticmethod
    def from_dict(item: Dict[str, Dict[str, str | float]]):
        """
        Instantiate a Normal Parameter from JSON

        Inputs:
            item (dict): Single dictionary with subdicts including the following information:
                            - UNITS (str)
                            - MEAN (float). Default to 0
                            - STDDEV (float). Default to 0

        Returns:
            NormalParameter
        """
        assert len(item.keys()) == 1, "More than 1 Parameter passed into JSON"
        key = list(item.keys())[0]

        # load in sub-dict to extract values
        subdict: Dict[str, str | float] = item.get(key)
        units = subdict.get("UNITS", "")
        mean = subdict.get("MEAN", 0)
        stddev = subdict.get("STDDEV", 0)

        # ensure default option for lazy JSON enters
        par = NormalParameter(key, units, mean, stddev)

        return par


class UniformParameter(Parameter):
    """
    Parameter to generate values following uniform distribution
    """

    def __init__(self, name: str, units: str, low: float, high: float):
        """
        Initialize Uniform Parameter

        Inputs:
            name (str): Name of parameter
            units (str): units of parameter
            low (float): lower limit of range of values
            high (float): upper limit of range of values
        """
        super().__init__(name, units)
        self._low = low
        self._high = high
        self._sym = "U"

    def __repr__(self) -> str:
        """
        Representation of Normal Parameter
        """
        return (
            "[Uniform Parameter] " + super().__repr__() + f"[{self._low}, {self._high}]"
        )

    def ideal(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] size array of values of central value between low and high

        Inputs:
            num (int): number of values to generate

        Returns:
            np.ndarray: array of "ideal" values
        """
        return 0.5 * (self._low + self._high) * np.ones(num)

    def modulate(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] size array of values following distribution set in init

        Inputs:
            num(int): number of values to generate

        Returns:
            np.ndarray: array of values following distribution

        """
        return np.random.uniform(self._low, self._high, num)

    def span(self, num: int = 1) -> np.ndarray:
        """
        Return [1 x n] array of values that span the bounds of the parameter

        Inputs:
            num (int)   : number of values to span the range

        Returns:
            np.ndarray  : array spanning the range
        """
        return np.linspace(self._low, self._high, num)


if __name__ == "__main__":
    d = {"A": {"MEAN": 24, "UNITS": "mm", "STDDEV": 0}}
    print(NormalParameter.from_dict(d))
