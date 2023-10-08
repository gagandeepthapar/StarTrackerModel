"""
plotter

Interface with composer to plot figures when prompted

Available models:
    - Standard
        - Histogram of camera performance
    - Verbose
        - Single row data, plots celestial sphere, focal plane, errors, etc

startrackermodel
"""

import logging
import logging.config

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, sim_data: pd.DataFrame):
        self.sim_data = sim_data

        # Create constants to reference
        self.celestial_sphere = self.get_sphere()

    def standard(self) -> None:
        """
        Plot all data in histograms

        Inputs:
            None

        Returns:
            None
        """

        pass

    def verbose_plot(self):
        """
        Plot process of modeling errors from sphere to errors to QUEST.
        Plots first row only

        Inputs:
            None

        Returns:
            None
        """
        pass

    @staticmethod
    def get_sphere(radius: float = 1) -> np.ndarray:
        """
        Generate surface data for a sphere of specified radius

        Inputs:
            radius (float): radius of sphere. Defaults to 1

        Returns:
            np.ndarray: [x, y, z] surface data of sphere
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)

        sphere_x = radius * np.outer(np.cos(theta), np.sin(phi))
        sphere_y = radius * np.outer(np.sin(theta), np.sin(phi))
        sphere_z = radius * np.outer(np.ones(theta.size), np.cos(phi))

        return np.array([sphere_x, sphere_y, sphere_z])

    @staticmethod
    def get_cone(
        ed_pt: np.ndarray,
        half_fov: float,
        start: np.ndarray = np.array([0, 0, 0]),
    ) -> np.ndarray:
        """
        Create surface data for a cone with specified end, start, half-angle

        Inputs:
            ed_pt (np.ndarray): center point of base of cone
            half_fov (float): half angle of FOV of cone [deg]
            start (np.ndarray): point of tip of cone. Default to [0,0,0]

        Returns:
            np.ndarray: surface data of cone
        """

        if np.array_equal(ed_pt, start):
            logger.warning(
                "End Point and Start Point are Identical. Moving Start to [0,0,0] and End to [1,0,0]"
            )
            ed_pt = np.array([1, 0, 0])
            start = np.array([0, 0, 0])

        cone_axis = ed_pt - start
        cone_height = np.linalg.norm(cone_axis)
        cone_rad = cone_height * np.tan(np.pi / 180.0 * half_fov)

        # construct flat data
        theta = np.linspace(0, 2 * np.pi, 100)
        flat_x = cone_rad * np.cos(theta)
        flat_y = cone_rad * np.sin(theta)
        flat_z = cone_height * np.ones(flat_x.shape)
        flat_data = np.array([flat_x, flat_y, flat_z])

        # construct rotm to rate cone into axis
        temp_vec = np.random.rand(3)
        temp_vec = temp_vec / np.linalg.norm(temp_vec)
        boresight_uvec = cone_axis / np.linalg.norm(cone_axis)

        # cross axis guaranteed to be orthogonal to boresight
        cross_axis = np.cross(boresight_uvec, temp_vec)
        cross_axis = cross_axis / np.linalg.norm(cross_axis)
        triad_axis = np.cross(boresight_uvec, cross_axis)
        triad_axis = triad_axis / np.linalg.norm(triad_axis)
        rotm = np.array([triad_axis, cross_axis, boresight_uvec]).T

        # create end point data
        cone_end = np.array(
            [rotm @ flat_data[:, i] for i in range(flat_data.shape[1])]
        ).T

        # create surface data
        cone_x = np.array([np.zeros(flat_data.shape[1]), cone_end[0, :]])
        cone_y = np.array([np.zeros(flat_data.shape[1]), cone_end[1, :]])
        cone_z = np.array([np.zeros(flat_data.shape[1]), cone_end[2, :]])
        return np.array([cone_x, cone_y, cone_z])


if __name__ == "__main__":
    axis = np.random.rand(3)
    axis = axis / np.linalg.norm(axis)
    cone = Plotter.get_cone(axis, 10)
    sphere = Plotter.get_sphere()

    logger.debug(axis)
    boresight = np.array([[0, 0, 0], axis])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(*sphere, color="whitesmoke", alpha=0.2)
    ax.plot_surface(*cone, color="r", alpha=0.5)
    ax.plot(*boresight.T)
    ax.scatter(*axis, marker="x")
    ax.axis("equal")
    # plt.show()
