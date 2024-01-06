"""
sensor

Model the image noise during image capture.

startrackermodel
"""
import logging
import logging.config

from typing import Dict

from data import CONSTANTS
from classes.component import Component
from classes.parameter import Parameter, UniformParameter, NormalParameter

import numpy as np
from numpy.matlib import repmat
import pandas as pd
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def comp_gaussian_kernel(window: int, std: float) -> np.ndarray:
    ax = np.linspace(-(window - 1) / 2, (window - 1) / 2, window)
    gauss = np.exp(-0.5 * ax**2 / std**2)
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


class Sensor(Component):
    # Constant parameters
    n_bits = 12
    v_max = 5
    v_min = 3.3
    q_e = 0.6
    max_e = 10_000
    gkern = comp_gaussian_kernel(5, 5)
    pixel_size = (5.7e-4) ** 2  # cm2 pixel area

    # Egap parameters
    E0 = 1.1557
    alpha = 7.021e-4
    beta = 1108

    def __init__(self, noise_cfg: Dict[str, Parameter]):
        """
        Initialize Software class
        """
        # self.photon_influx = noise_cfg.get(
        #     "PHOTON_INFLUX",
        #     NormalParameter("PHOTON_INFLUX", "photoelectrons/s/mm2", 19100, 0),
        # )
        # self.int_time = noise_cfg.get(
        #     "INTEGRATION_TIME", NormalParameter("INTEGRATION_TIME", "sec", 0.2, 0)
        # )
        # self.aperture_dia = noise_cfg.get(
        #     "APERTURE_DIAMETER", NormalParameter("APERTURE_DIAMETER", "mm", 7.75, 0)
        # )
        # self.darknoise = noise_cfg.get(
        #     "DARKNOISE_FIGURE", NormalParameter("DARKNOISE_FIGURE", "-", 0, 0)
        # )

        self.mag_threshold = noise_cfg.get(
            "MAG_DETECTION_THRESHOLD",
            NormalParameter("MAG_DETECTION_THRESHOLD", "Mv", 1_000_000, 0),
        )

        self.object_list = [self.mag_threshold]
        return

    @staticmethod
    def compute_neighboring_pixels(x: float, y: float) -> np.ndarray:
        return np.array(
            [
                [np.floor(x), np.floor(y)],
                [np.floor(x), np.ceil(y)],
                [np.ceil(x), np.floor(y)],
                [np.ceil(x), np.ceil(y)],
            ]
        ).astype(int)

    @staticmethod
    def compute_photoelectrons(
        star_mag: float, photon_influx: float, int_time: float, aperture_diameter: float
    ) -> float:
        return (
            Sensor.q_e
            * photon_influx
            * 1
            / (2.5**star_mag)
            * int_time
            * np.pi
            * aperture_diameter**2
            / 4
        )

    @staticmethod
    def compute_brightness_threshold(
        fx: int,
        fy: int,
        s_set: np.ndarray,
        photoelec: np.ndarray,
        temp: float,
        int_time: float,
        darknoise_figure: float,
    ) -> float:
        fx = int(fx)
        fy = int(fy)
        # initialize empty sensor
        scene = np.zeros((fx, fy))
        s_set = np.delete(s_set, 2, 1)

        # modify s_set to inhabit multiple cells
        star_bounds = np.zeros((4 * len(s_set), 2))
        for i in range(len(s_set)):
            bounds = Sensor.compute_neighboring_pixels(*(s_set[i]))
            diffs = 1 - np.abs(bounds - s_set[i])
            norms = np.apply_along_axis(
                np.linalg.norm, 1, diffs
            )  # get distance between each pixel and true centroid
            dist_photoelec = (
                norms * photoelec[i] / norms.sum()
            )  # compute distribution of photoelectrons
            # bounds = s_set[i]
            dist_photoelec = photoelec[i]
            scene[
                bounds[:, 0], bounds[:, 1]
            ] += dist_photoelec  # place distributed photoelectons onto sensor
            star_bounds[4 * i : 4 * i + 4, :] = bounds

        star_bounds = star_bounds.astype(int)

        # shot noise
        scene = np.random.poisson(scene)

        # dark noise
        Egap = Sensor.E0 - Sensor.alpha * temp**2 / (temp + Sensor.beta)
        darknoise = (
            Sensor.pixel_size
            * darknoise_figure
            * temp**1.5
            * np.exp(-Egap / (2 * CONSTANTS.BOLTZMANN * temp))
        ) ** 0.5
        # logger.critical(darknoise)
        # logger.critical(darknoise_figure)
        # dn = np.abs(np.random.normal(0, (darknoise) * 100, scene.shape) * int_time)
        dn = np.ones(scene.shape) * 1 * darknoise_figure * int_time
        dn = np.random.poisson(dn)

        # limitation
        scene = scene + dn
        scene[np.unravel_index(np.argmin(scene, axis=None), scene.shape)] = 0
        scene = np.maximum(scene, 0)
        scene = np.minimum(scene, Sensor.max_e)

        # plt.imshow(scene)
        # plt.show()
        # raise ValueError

        # conv to voltage
        sense = (Sensor.v_max - Sensor.v_min) / Sensor.max_e
        reset_noise = np.sqrt(CONSTANTS.BOLTZMANN * temp / (1 / sense))
        v_reset = np.exp(np.random.normal(0, reset_noise, scene.shape)) - 1
        scene_volts = Sensor.v_min + v_reset - (scene * sense)

        scene_volts = np.floor(scene_volts)

        # Conversion to digital signal
        Aadc = 2**Sensor.n_bits / (Sensor.v_max - Sensor.v_min)
        Dn = Aadc * Sensor.v_max / Sensor.max_e * scene
        Dn = np.round(Dn)
        # plt.imshowDn)
        # plt.show()

        kern = convolve2d(Dn, Sensor.gkern, mode="same")
        star_bounds_flag = (
            kern[star_bounds[:, 0], star_bounds[:, 1]] >= kern.mean() + 5 * kern.std()
        )
        flags = np.apply_along_axis(
            np.all, 1, (np.split(star_bounds_flag, len(star_bounds_flag) // 4))
        )

        # print(kern[star_bounds[:, 0], star_bounds[:, 1]])
        # plt.imshow(kern)
        # plt.show()
        # logger.critical(kern.mean() + 5 * kern.std())
        # kern[kern < (kern.mean() + 5 * kern.std())] = 0

        return flags

    def modulate(self, num: int) -> pd.DataFrame:
        """
        Update modulate base class with derived parameters

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF
        """
        return super().modulate(num)

    def span(self, num: int) -> pd.DataFrame:
        """
        Update modulate base class with derived parameters

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF
        """
        return super().span(num)

    def __repr__(self) -> str:
        """
        Return representation of class
        """
        return f"Sensor Class:\n{super().__repr__()}"

    @staticmethod
    def plotimg(matrix, title):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(matrix)
        matrix[np.unravel_index(np.argmin(matrix), matrix.shape)] = 0
        ax.set_title(title)
        ax.axis("equal")
        plt.show()

    @staticmethod
    def signalimg(N: int):
        fx, fy = 100, 100
        ti = 0.2
        A = 15 * np.pi
        prnu = 0.5 / 100
        dfpn = 1 / 100
        sigsf = 5
        sigreset = np.sqrt(3)
        sigcol = 0.1 / 100
        temp = 300

        s_set = np.array([np.random.uniform(0, 99, 2) for _ in range(N)]).astype(int)
        mag = np.random.uniform(0, 7, N)
        pe = 19100 / 0.6 * 1 / (2.5**mag) * ti * A

        signal = np.zeros((fx, fy))
        signal[s_set[:, 0], s_set[:, 1]] = pe
        Sensor.plotimg(signal, "raw signal")

        signal_pois = np.random.poisson(signal)
        signal_pois = signal_pois * 0.6
        Sensor.plotimg(signal_pois, "pois rand signal")

        prnu = np.multiply(signal_pois, np.random.normal(0, prnu, signal_pois.shape))
        signal_prnu = signal_pois + prnu
        Sensor.plotimg(signal_prnu, "prnu addition")

        Egap = Sensor.E0 - Sensor.alpha * temp**2 / (temp + Sensor.beta)
        Dr = (
            Sensor.pixel_size
            * 1
            * temp**1.5
            * np.exp(-Egap / (2 * CONSTANTS.BOLTZMANN * temp))
        )

        dcshot = np.random.poisson(np.ones(signal_prnu.shape) * Dr * ti)
        dfpn = dcshot + np.multiply(dcshot, np.random.lognormal(0, ti * Dr * dfpn))
        print(dfpn.max())
        Sensor.plotimg(dfpn, "only darknoise")

        Isf = np.abs(np.random.normal(0, sigsf))
        Isf = np.round(Isf)

        Ilight = signal_prnu
        Idark = dcshot + Isf
        Itotal = np.round(Ilight + Idark)
        Sensor.plotimg(Itotal, "total noise electron")

        Ireset = np.random.lognormal(0, sigreset, Ilight.shape)
        Iref = np.ones(Ilight.shape) * Sensor.v_min + 0
        Sensor.plotimg(Ireset, "Reset Noise")

        Asn = 2**Sensor.n_bits * (Sensor.v_max - Sensor.v_min) / 33000
        Vlight = Sensor.v_max - (Ilight) * 5
        Sensor.plotimg(Vlight, "V light")

        coloffset = np.random.normal(0, sigcol, fy)
        coloffset = repmat(coloffset, 1, fx)
        coloffset = coloffset.T

        return


if __name__ == "__main__":
    n = 50
    snes = Sensor({})
    snes.signalimg(10)
    # P[0] = 100
