"""
composer

Interface with driver module to setup and run monte carlo

Steps to model:
    - Generate unique star tracker data from inputs
    - Generate unique environmental data from random distribution 
    - Determine visible stars and apply camera distortions 
    - Compare true VS measured quaternions 
    - Store results
    - Plot data

startrackermodel
"""

from functools import total_ordering
import logging
import logging.config
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import CONSTANTS
from classes.hardware import Hardware
from classes.software import Software
from classes.sensor import Sensor
from classes.environment import Environment
from classes.attitude import Attitude
from classes.parameter import Parameter
from classes.component import Component
from classes.starframe import StarFrame
from classes.enums import SimType

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Composer:
    """
    Converts user-supplied arguments into actionable DataFrames
    """

    def __init__(
        self,
        hardware: Hardware,
        software: Software,
        sensor: Sensor,
        environment: Environment,
        sim_type: SimType,
    ):
        self.hardware = hardware
        self.software = software
        self.sensor = sensor
        self.environment = environment
        self.attitude = Attitude()
        self.sim_type = sim_type

        self.components: List[Component] = [
            self.attitude,
            self.hardware,
            self.software,
            self.sensor,
            self.environment,
        ]

        self.starframe = StarFrame()
        self.model_data = pd.DataFrame()

    def generate_data(self, num: int, param_select: List[str] = None) -> pd.DataFrame:
        """
        Generate sim-specific data and store in class

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: Dataframe of data
        """

        data_df = pd.DataFrame()
        match self.sim_type:
            case SimType.MONTE_CARLO:
                data_df = self.modulate(num)

            case SimType.SENSITIVITY:
                data_df = self.span(num, param_select)

        self.model_data = data_df
        return data_df

    def modulate(self, num: int) -> pd.DataFrame:
        """
        Generate all data based on user-supplied information

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: Df of all columns randomized according to distribution
        """
        data_df = pd.concat([comp.modulate(num) for comp in self.components], axis=1)
        return data_df

    def span(self, num: int, param_select: List[str] = None) -> pd.DataFrame:
        """
        Generate Sensitivity Analysis inputs

        Inputs:
            num (int): number of rows to generate

        Returns:
            pd.DataFrame: DF of Sensitivity analysis data
        """
        # extract all relevant parameters from child components
        total_objects: List[Parameter] = []
        for comp in self.components[1:]:
            total_objects.extend(comp.object_list)

        if param_select is None:
            param_select = [param.name for param in total_objects]

        # span each parameter based on total number of parameters
        # data_frame = pd.DataFrame()
        param_df = pd.DataFrame(
            {
                (mod_param.name): (
                    mod_param.span(num)
                    if mod_param.name in param_select
                    else mod_param.ideal(num)
                )
                for mod_param in total_objects
            }
        )
        # data_frame.reset_index(inplace=True)
        data_frame = pd.concat(
            [self.attitude.modulate(len(param_df.index)), param_df],
            axis=1,
        )
        return data_frame

    def run_model(self):
        """
        HARDWARE PERTURBATION SIMULATION
        """
        # simulate temperatuer
        self.model_data["TEMPERATURE"] = self.model_data[["EMISSIVITY"]].apply(
            lambda row: np.random.uniform(*Environment.temp_trends(row.EMISSIVITY))
            + CONSTANTS.C_TO_K,
            axis=1,
        )
        # update focal length based on thermal coefficient
        self.model_data["DT_FOCAL_LENGTH"] = self.model_data[
            [
                "FOCAL_LENGTH",
                "FOCAL_ARRAY_DELTA_Z",
                "TEMPERATURE",
                "OPTO_THERMAL_COEFFICIENT",
            ]
        ].apply(
            lambda row: (row.TEMPERATURE - CONSTANTS.C_TO_K)
            * row.OPTO_THERMAL_COEFFICIENT
            * (row.FOCAL_LENGTH + row.FOCAL_ARRAY_DELTA_Z),
            axis=1,
        )

        # compute hardware params
        self.model_data["MAX_FOV"] = self.model_data[
            [
                "FOCAL_ARRAY_X",
                "FOCAL_ARRAY_Y",
                "FOCAL_LENGTH",
                "DT_FOCAL_LENGTH",
                "FOCAL_ARRAY_DELTA_Z",
            ]
        ].apply(
            lambda row: Hardware.compute_fov(
                np.linalg.norm([row.FOCAL_ARRAY_X, row.FOCAL_ARRAY_Y]),
                row.FOCAL_LENGTH + row.DT_FOCAL_LENGTH + row.FOCAL_ARRAY_DELTA_Z,
            ),
            axis=1,
        )

        self.model_data["R_GAMMA_PI"] = self.model_data[
            ["FOCAL_ARRAY_THETA_Z", "FOCAL_ARRAY_THETA_Y", "FOCAL_ARRAY_THETA_X"]
        ].apply(
            lambda row: Attitude.rotm_z(CONSTANTS.DEG2RAD * row.FOCAL_ARRAY_THETA_Z)
            @ Attitude.rotm_y(CONSTANTS.DEG2RAD * row.FOCAL_ARRAY_THETA_Y)
            @ Attitude.rotm_x(CONSTANTS.DEG2RAD * row.FOCAL_ARRAY_THETA_X),
            axis=1,
        )

        self.model_data["R_F_GAMMA_GAMMA"] = self.model_data[
            [
                "FOCAL_LENGTH",
                "DT_FOCAL_LENGTH",
                "FOCAL_ARRAY_DELTA_X",
                "FOCAL_ARRAY_DELTA_Y",
                "FOCAL_ARRAY_DELTA_Z",
                "R_GAMMA_PI",
            ]
        ].apply(
            lambda row: Hardware.get_r_f_gamma_gamma(
                row.FOCAL_LENGTH,
                np.array(
                    [
                        row.FOCAL_ARRAY_DELTA_X,
                        row.FOCAL_ARRAY_DELTA_Y,
                        row.FOCAL_ARRAY_DELTA_Z + row.DT_FOCAL_LENGTH,
                    ]
                ),
                row.R_GAMMA_PI,
            ),
            axis=1,
        )

        # store star within fov per scenario
        self.model_data["STAR_V_SET"], self.model_data["STAR_MAG"] = zip(
            *self.model_data[["UVEC_ECI", "MAX_FOV"]].apply(
                lambda row: self.starframe.filter_stars(row.UVEC_ECI, row.MAX_FOV),
                axis=1,
            )
        )

        # get stars in body frame via R_STAR
        self.model_data["STAR_W_SET"] = self.model_data[["R_STAR", "STAR_V_SET"]].apply(
            lambda row: np.array(
                [row.R_STAR @ star_v_i for star_v_i in row.STAR_V_SET]
            ),
            axis=1,
        )

        # compute stars on rotated field
        self.model_data["STAR_W_ROT"] = self.model_data[
            ["R_GAMMA_PI", "STAR_W_SET"]
        ].apply(
            lambda row: np.array([row.R_GAMMA_PI @ w_i for w_i in row.STAR_W_SET]),
            axis=1,
        )

        # compute image coordinates of incident star
        self.model_data["LAMBDA_STAR"] = self.model_data[
            ["STAR_W_ROT", "R_F_GAMMA_GAMMA"]
        ].apply(
            lambda row: np.array(
                [-row.R_F_GAMMA_GAMMA[2] / wi_rot[2] for wi_rot in row.STAR_W_ROT]
            ),
            axis=1,
        )

        # compute s hash set
        self.model_data["STAR_S_HASH_SET"] = self.model_data[
            ["LAMBDA_STAR", "STAR_W_ROT", "R_F_GAMMA_GAMMA"]
        ].apply(
            lambda row: np.array(
                [
                    star_lam_star * wi_rot + row.R_F_GAMMA_GAMMA
                    for (star_lam_star, wi_rot) in zip(row.LAMBDA_STAR, row.STAR_W_ROT)
                ]
            ),
            axis=1,
        )

        # remove stars outside sensor
        self.model_data["IN_SENSOR"] = self.model_data[
            ["FOCAL_ARRAY_X", "FOCAL_ARRAY_Y", "STAR_S_HASH_SET"]
        ].apply(
            lambda row: np.array(
                [
                    np.all(
                        np.less_equal(
                            np.abs(coords),
                            np.array([row.FOCAL_ARRAY_X / 2, row.FOCAL_ARRAY_Y / 2, 0]),
                        )
                    )
                    for coords in row.STAR_S_HASH_SET
                ]
            ),
            axis=1,
        )

        self.model_data["STAR_V_SET"] = self.model_data[
            ["STAR_V_SET", "IN_SENSOR"]
        ].apply(lambda row: row.STAR_V_SET[row.IN_SENSOR], axis=1)

        self.model_data["STAR_MAG"] = self.model_data[["STAR_MAG", "IN_SENSOR"]].apply(
            lambda row: row.STAR_MAG[row.IN_SENSOR], axis=1
        )

        self.model_data["STAR_S_HASH_SET"] = self.model_data[
            ["STAR_S_HASH_SET", "IN_SENSOR"]
        ].apply(lambda row: row.STAR_S_HASH_SET[row.IN_SENSOR], axis=1)

        # store number of stars in each scene
        self.model_data["N_STARS"] = self.model_data["STAR_MAG"].apply(
            lambda row: len(row)
        )

        # remove sims with insufficient number of stars
        self.model_data = self.model_data[self.model_data.N_STARS >= 2]

        """ 
        IMAGE NOISE SIMULATION
        """
        self.model_data["DISCERNIBLE"] = self.model_data[
            ["STAR_MAG", "MAG_DETECTION_THRESHOLD"]
        ].apply(
            lambda row: row.STAR_MAG <= row.MAG_DETECTION_THRESHOLD,
            axis=1,
        )

        self.model_data["STAR_V_SET"] = self.model_data[
            ["STAR_V_SET", "DISCERNIBLE"]
        ].apply(lambda row: row.STAR_V_SET[row.DISCERNIBLE], axis=1)

        self.model_data["STAR_MAG"] = self.model_data[
            ["STAR_MAG", "DISCERNIBLE"]
        ].apply(lambda row: row.STAR_MAG[row.DISCERNIBLE], axis=1)

        self.model_data["STAR_S_HASH_SET"] = self.model_data[
            ["STAR_S_HASH_SET", "DISCERNIBLE"]
        ].apply(lambda row: row.STAR_S_HASH_SET[row.DISCERNIBLE], axis=1)

        # store number of stars in each scene
        self.model_data["N_STARS"] = self.model_data["STAR_MAG"].apply(
            lambda row: len(row)
        )

        # remove sims with insufficient number of stars
        self.model_data = self.model_data[self.model_data.N_STARS >= 2]

        """ 
        CENTROIDING ERROR SIMULATION
        """
        self.model_data["ERROR_DIR"] = self.model_data["N_STARS"].apply(
            lambda n: np.random.uniform(0, 2 * np.pi, n)
        )
        self.model_data["C_DEL_X"], self.model_data["C_DEL_Y"] = zip(
            *self.model_data[["ERROR_DIR", "ERROR_MULTIPLIER", "STAR_MAG"]].apply(
                lambda row: Software.base_acc(row.STAR_MAG)
                * row.ERROR_MULTIPLIER
                * (np.cos(row.ERROR_DIR), np.sin(row.ERROR_DIR)),
                axis=1,
            )
        )

        # recompute measured centroid
        self.model_data["STAR_S_HASH_SET"] = self.model_data[
            ["STAR_S_HASH_SET", "C_DEL_X", "C_DEL_Y"]
        ].apply(
            lambda row: np.array(
                [
                    s_hash_i + np.array([row.C_DEL_X[i], row.C_DEL_Y[i], 0])
                    for (i, s_hash_i) in enumerate(row.STAR_S_HASH_SET)
                ]
            ),
            axis=1,
        )

        # recompute measured unit vectors
        self.model_data["STAR_W_HASH_SET"] = self.model_data[
            ["STAR_S_HASH_SET", "FOCAL_ARRAY_X", "FOCAL_ARRAY_Y", "FOCAL_LENGTH"]
        ].apply(
            lambda row: np.array(
                [
                    Attitude.unit(np.array([0, 0, row.FOCAL_LENGTH]) - s_hash_i)
                    for s_hash_i in row.STAR_S_HASH_SET
                ]
            ),
            axis=1,
        )

        # determine failed star identifications
        self.model_data["STAR_IDENT"] = self.model_data["N_STARS"].apply(
            lambda n: np.array([np.random.uniform(0.0, 1.0, n)])
        )

        self.model_data["IDENTIFIED"] = self.model_data[
            ["STAR_IDENT", "IDENTIFICATION_RATE"]
        ].apply(
            lambda row: np.array(
                [star_ident < row.IDENTIFICATION_RATE for star_ident in row.STAR_IDENT]
            ).flatten(),
            axis=1,
        )  # True values are stars that were identified

        self.model_data["STAR_V_SET"] = self.model_data[
            ["STAR_V_SET", "IDENTIFIED"]
        ].apply(lambda row: row.STAR_V_SET[row.IDENTIFIED], axis=1)

        self.model_data["STAR_MAG"] = self.model_data[["STAR_MAG", "IDENTIFIED"]].apply(
            lambda row: row.STAR_MAG[row.IDENTIFIED], axis=1
        )

        self.model_data["STAR_S_HASH_SET"] = self.model_data[
            ["STAR_S_HASH_SET", "IDENTIFIED"]
        ].apply(lambda row: row.STAR_S_HASH_SET[row.IDENTIFIED], axis=1)

        self.model_data["STAR_W_HASH_SET"] = self.model_data[
            ["STAR_W_HASH_SET", "IDENTIFIED"]
        ].apply(lambda row: row.STAR_W_HASH_SET[row.IDENTIFIED], axis=1)

        # store number of stars in each scene
        self.model_data["N_STARS"] = self.model_data["STAR_MAG"].apply(
            lambda row: len(row)
        )

        # remove sims with insufficient number of stars
        self.model_data = self.model_data[self.model_data.N_STARS >= 2]

        # drop intermediate columns
        self.model_data = self.model_data.drop(
            ["STAR_W_ROT", "LAMBDA_STAR", "IN_SENSOR", "ERROR_DIR"],
            axis=1,
        )

        """ 
        AGGREGATE EFFECT COMPARISON
        """
        # compute quaternion estimate
        self.model_data["Q_HASH"] = self.model_data[
            ["STAR_V_SET", "STAR_W_HASH_SET"]
        ].apply(
            lambda row: Attitude.quest_algorithm(
                row.STAR_W_HASH_SET,
                np.array([star_hash_i for star_hash_i in row.STAR_V_SET]),
            ),
            axis=1,
        )

        # compute quaternion error
        self.model_data["ANGULAR_ERROR"] = self.model_data[["Q_STAR", "Q_HASH"]].apply(
            lambda row: CONSTANTS.RAD2ARCSEC
            * Attitude.quat_compare(row.Q_STAR, row.Q_HASH),
            axis=1,
        )
