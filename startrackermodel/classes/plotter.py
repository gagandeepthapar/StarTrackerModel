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
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from data import CONSTANTS

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, sim_data: pd.DataFrame):
        self.sim_data = sim_data
        self.verbose_data = self.sim_data.iloc[0]
        self.boresight = np.array([[0, 0, 0], self.verbose_data.UVEC_ECI]).T
        self.stars = [
            np.array([[0, 0, 0], star]).T for star in self.verbose_data.STAR_V_SET
        ]

        # Create constants to reference
        self.celestial_sphere = self.get_sphere()
        self.fov_cone = self.get_cone(
            self.boresight[:, 1], CONSTANTS.RAD2DEG * self.verbose_data.MAX_FOV / 2
        )

    def show(self) -> None:
        """
        Show Plots
        """
        plt.show()
        return

    def standard(self) -> None:
        """
        Plot all data in histograms

        Inputs:
            None

        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot()

        xrange = np.linspace(0, self.sim_data.ANGULAR_ERROR.max() * 1.1)
        # mean = self.sim_data.ANGULAR_ERROR.mean()
        std = self.sim_data.ANGULAR_ERROR.std()
        yrange = (
            np.sqrt(2)
            / (std * np.sqrt(np.pi))
            * np.exp(-(xrange**2) / (2 * std**2))
        )
        nrange = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xrange) / std) ** 2)

        ax.hist(
            self.sim_data["ANGULAR_ERROR"],
            density=True,
            label="Monte Carlo Distribution",
        )
        ax.plot(xrange, yrange, "r--", label="Half-Normal Distribution")
        ax.plot(xrange, nrange, "g--", label="Normal Distribution")
        ax.set_xlabel("Angular Error [arcsec]")
        ax.set_ylabel("Probability Density")
        ax.set_title("Probability Density Function for Star Tracker Accuracy")
        ax.legend()

        return

    def verbose_plot(self):
        """
        Plot process of modeling errors from sphere to errors to QUEST.
        Plots first row only

        Inputs:
            None

        Returns:
            None
        """
        plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        self.plot_star_sphere(ax)

        ax = fig.add_subplot(gs[:, 1], projection="3d")
        # self.plot_gamma_plate(ax)

        ax = fig.add_subplot(gs[1, 0])
        self.plot_star_scene(ax)

        return

    def plot_star_sphere(self, ax: Axes3D):
        ax.plot_surface(*self.celestial_sphere, alpha=0.1)
        ax.plot_surface(*self.fov_cone, alpha=0.5)
        ax.plot(*self.boresight, "k")
        ax.scatter(*self.boresight[:, 1], marker="x")
        [ax.plot(*star) for star in self.stars]

        ax.axis("equal")
        ttl = f"RA: {CONSTANTS.RAD2DEG*self.sim_data.RIGHT_ASCENSION:.3f}, DEC: {CONSTANTS.RAD2DEG*self.sim_data.DECLINATION:.3f}, ROLL: {CONSTANTS.RAD2DEG*self.sim_data.ROLL:.3f}"
        ax.set_title(ttl)

        return

    def plot_star_scene(self, ax: Axes):
        fx = self.sim_data.FOCAL_ARRAY_X
        fy = self.sim_data.FOCAL_ARRAY_Y

        rect = Rectangle(
            (-fx / 2, -fy / 2), fx, fy, alpha=1, color="whitesmoke", label=None
        )
        ax.add_patch(rect)
        ax.scatter(
            self.boresight[1, 0],
            self.boresight[1, 1],
            marker="x",
            color="k",
            label="Boresight",
        )
        for i in range(len(self.sim_data.STAR_S_HASH_SET)):
            if i == 1:
                label_leg_B = "Perturbed Star Position"
            else:
                label_leg_B = None

            h_i = self.sim_data.STAR_S_HASH_SET[i]

            ax.scatter(h_i[0], h_i[1], color="k", marker=".", label=label_leg_B)

        ax.axis("equal")
        ax.legend()

        return

    def plot_gamma_plate(self, ax):
        (
            th_x,
            th_y,
            th_z,
        ) = (
            self.sim_data.FOCAL_ARRAY_THETA_X,
            self.sim_data.FOCAL_ARRAY_THETA_Y,
            self.sim_data.FOCAL_ARRAY_THETA_Z,
        )
        dx, dy, dz = (
            self.sim_data.FOCAL_ARRAY_DELTA_X,
            self.sim_data.FOCAL_ARRAY_DELTA_Y,
            self.sim_data.FOCAL_ARRAY_DELTA_Z,
        )
        axd, ayd = self.sim_data.FOCAL_ARRAY_X, self.sim_data.FOCAL_ARRAY_Y
        fx = self.sim_data.FOCAL_LENGTH

        F = np.array([0, 0, fx])

        S = fx * self.sim_data.STAR_W_SET[0] + F

        arx = np.linspace(-axd / 2, axd / 2, 101)
        ary = np.linspace(-ayd / 2, ayd / 2, 101)
        arz = np.linspace(0, 0, arx.size)

        R_gp = self.sim_data.R_GAMMA_PI

        arx, ary = np.meshgrid(arx, ary)
        arz = np.zeros(arx.shape)

        gxp = np.zeros(arx.shape)
        gyp = np.zeros(ary.shape)
        gzp = np.zeros(arz.shape)

        for i in range(arx.shape[0]):
            for j in range(arx.shape[1]):
                v = np.array([arx[i, j], ary[i, j], arz[i, j]]) + np.array([dx, dy, dz])
                vg = R_gp @ v
                gxp[i, j], gyp[i, j], gzp[i, j] = vg[0], vg[1], vg[2]

        cx = np.mean(gxp)
        cy = np.mean(gyp)
        cz = np.mean(gzp)

        lam_i = -fx / ((S - F)[2])
        Si = lam_i * (S - F) + F

        rsg = R_gp @ (F + np.array([dx, dy, dz]))
        lam_hash = -(rsg)[2] / (R_gp @ (S - F))[2]
        Shash = R_gp.T @ (lam_hash * (R_gp @ (S - F)) + rsg) - np.array([dx, dy, dz])
        Shash = 2 * Si - Shash

        S_i = np.array([Si, S])
        # S_i = S_i.T
        S_hash = np.array([Shash, S])
        # S_hash = S_hash.T

        base_p = np.eye(3)
        base_g = np.zeros((3, 3))
        for i in range(3):
            base_g[i, :] = R_gp @ (base_p[i, :])

        base_g = base_g.T

        m = 125
        base_p = base_p * m
        base_g = base_g * m

        c = ["r", "g", "b"]
        rp = [r"$\Pi_x$", r"$\Pi_z$", r"$\Pi_z$"]
        rg = [r"$\Gamma_x$", r"$\Gamma_z$", r"$\Gamma_z$"]

        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # Hide grid lines
        # ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis("off")
        plt.grid(b=False)
        ax.scatter(*S, color="goldenrod", label=r"$\mathbf{S}$", marker=r"$*$")
        ax.scatter(0, 0, fx, color="k", label=r"$\mathbf{F}$")
        ax.plot_surface(
            arx, ary, arz, alpha=0.1, color="blue", linewidth=0, label=r"$\Pi$"
        )
        for i in range(3):
            ax.plot(
                [0, base_p[0, i]],
                [0, base_p[1, i]],
                [0, base_p[2, i]],
                color=c[i],
                linewidth=2,
                label=rp[i],
            )

        ax.plot_surface(
            gxp, gyp, gzp, alpha=0.1, color="red", linewidth=0, label=r"$\Gamma$"
        )
        for i in range(3):
            ax.plot(
                cx + [0, base_g[0, i]],
                cy + [0, base_g[1, i]],
                cz + [0, base_g[2, i]],
                color=c[i],
                linestyle="--",
                linewidth=2,
                label=rg[i],
            )

        ax.scatter(*Si, marker="x", color="blue", label=r"$S_i$")
        ax.scatter(*Shash, marker="x", label=r"$S_i^\#$", color="red")
        ax.plot(S_i[:, 0], S_i[:, 1], S_i[:, 2], label=r"$\mathbf{w_i}$", color="m")
        ax.plot(
            S_hash[:, 0],
            S_hash[:, 1],
            S_hash[:, 2],
            label=r"$\mathbf{w_i^\#}$",
            color="m",
            linestyle="--",
        )
        ax.plot([0, 0], [0, 0], [0, fx], "k--", label=r"$\mathbf{r}_{F/\Pi, \Pi}$")
        ax.scatter(0, 0, 0, color="k", marker="o", label=r"$(x_0, y_0)$")
        ax.scatter(-axd / 2, 0, 0, marker="x")
        ax.plot(
            [(cx), 0], [(cy), 0], [cz, fx], "r--", label=r"$\mathbf{r}_{F/\Gamma, \Pi}$"
        )
        ax.plot(
            [0, cx], [0, cy], [0, cz], label=r"$\mathbf{r}_{\Gamma/\Pi,\Pi}$", color="k"
        )
        ax.axis("equal")
        ax.legend()
        return

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
