"""
plotter Interface with composer to plot figures when prompted
Available models:
    - Standard
        - Histogram of camera performance
    - Verbose
        - Single row data, plots celestial sphere, focal plane, errors, etc

startrackermodel
"""

import logging
import logging.config
from typing import List
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.special import jn, jn_zeros
from data import CONSTANTS
from classes.attitude import Attitude
from classes.starframe import StarFrame
from copy import deepcopy

logging.config.dictConfig(CONSTANTS.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, sim_data: pd.DataFrame):
        self.catalog = StarFrame.prep_star_catalog()
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

    def star_img(self, ra: float, dec: float):
        (
            w,
            h,
        ) = (
            1024,
            1024,
        )
        F = 3500

        bs = Attitude.ra_dec_to_uvec(ra, dec)
        mfov = np.arctan(np.linalg.norm([w / 2, h / 2]) / F)
        print(mfov)
        self.catalog["INFOV"] = self.catalog[["UVEC_ECI"]].apply(
            lambda row: np.arccos(bs.dot(row.UVEC_ECI)) <= mfov, axis=1
        )

        self.catalog = self.catalog[self.catalog["INFOV"]]
        zeb = bs
        xeb = np.array([np.cos(ra - np.pi / 2), np.sin(ra - np.pi / 2), 0])
        yeb = np.cross(zeb, xeb)
        Teb = np.array([xeb, yeb, zeb])

        self.catalog["Wstar"] = self.catalog[["UVEC_ECI"]].apply(
            lambda row: Teb @ row.UVEC_ECI, axis=1
        )
        self.catalog["Lstar"] = self.catalog[["Wstar"]].apply(
            lambda row: -F / row.Wstar[2], axis=1
        )
        self.catalog["COORDS"] = self.catalog[["Wstar", "Lstar"]].apply(
            lambda row: np.round(row.Lstar * row.Wstar + np.array([0, 0, F])), axis=1
        )

        self.catalog["INARR"] = self.catalog[["COORDS"]].apply(
            lambda row: np.abs(row.COORDS[0]) <= w / 2
            and np.abs(row.COORDS[1]) <= h / 2,
            axis=1,
        )

        self.catalog = self.catalog[self.catalog.INARR]
        self.catalog["PHOTON"] = self.catalog[["v_magnitude"]].apply(
            lambda row: 19100
            / 0.63
            / (2.5**row.v_magnitude)
            * 0.2
            * 7.5**2
            * np.pi,
            axis=1,
        )

        image = np.zeros((w, h))
        for i, row in self.catalog.iterrows():
            x = int(row.COORDS[0] + w / 2)
            y = int(row.COORDS[1] + h / 2)
            image[x, y] = row.PHOTON

        image = np.random.poisson(image)
        image = 0.63 * image

        Ego = 1.1557
        beta = 1108
        alpha = 7.021e-4
        Eg = Ego - alpha * 300**2 / (300 + beta)
        k = CONSTANTS.BOLTZMANN
        Dr = 3.5e-4 * 50000 * 300**1.5 * np.exp(-Eg / (2 * k * 300))
        Idr = np.ones(image.shape) * Dr
        Idr = np.random.poisson(Idr)

        print(Idr.mean())
        print(Dr)
        image = image + Idr

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(image)

        plt.show()

        return

    def scene(self, ra: float, dec: float, roll: float = np.pi / 4):
        plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
        bs = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])

        bs_proj = 1 * bs
        bs_proj[2] = 0
        bs_proj = bs_proj / np.linalg.norm(bs_proj)

        r = 0.5
        theta1 = np.linspace(0, ra, 100)
        x1 = r * np.cos(theta1)
        y1 = r * np.sin(theta1)
        z1 = r * np.zeros(x1.shape)

        theta2 = np.linspace(0, dec, 100)
        y2 = r * np.cos(theta2)
        z2 = r * np.sin(theta2)
        x2 = np.zeros(y2.shape)
        circ = np.array([x2, y2, z2]).T
        circ2 = np.zeros(circ.shape)
        for i in range(len(x1)):
            circ2[i, :] = Attitude.rotm_z(-(np.pi / 2 - ra)) @ circ[i, :]

        circ2 = circ2.T

        alphatext = np.array([np.cos(ra / 2), np.sin(ra / 2), 0])
        alphatext = alphatext / np.linalg.norm(alphatext)

        dectext = np.array(
            [
                np.cos(ra) * np.cos(dec / 2),
                np.sin(ra) * np.cos(dec / 2),
                np.sin(dec / 2),
            ]
        )
        dectext = dectext / np.linalg.norm(dectext)

        fovhalf = 7.5
        cone = Plotter.get_cone(bs, fovhalf)

        self.catalog["INFOV"] = self.catalog[["UVEC_ECI"]].apply(
            lambda row: np.arccos(bs.dot(row.UVEC_ECI)) <= np.pi / 180 * fovhalf, axis=1
        )
        filtcat = self.catalog[self.catalog.INFOV]
        filtcat = filtcat[filtcat.v_magnitude <= 5.5]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.plot_surface(*self.celestial_sphere, alpha=0.1)
        ax.plot_surface(*cone, alpha=0.15, color="purple")

        basis = np.eye(3)
        xb = np.array([[0, 0, 0], basis[0, :]]).T
        yb = np.array([[0, 0, 0], basis[1, :]]).T
        zb = np.array([[0, 0, 0], basis[2, :]]).T

        ax.plot(*xb, "r")
        ax.plot(*yb, "g")
        ax.plot(*zb, "b")

        lam = 1.2
        ax.text(*(lam**2 * basis[0, :]), r"$\hat{x}_V$", fontsize="xx-large")
        ax.text(*(lam * basis[1, :]), r"$\hat{y}_V$", fontsize="xx-large")
        ax.text(*(lam * basis[2, :]), r"$\hat{z}_V$", fontsize="xx-large")
        ax.text(*(0.6 * lam * alphatext), r"$\alpha$", fontsize="xx-large")
        ax.text(*(0.6 * lam * dectext), r"$\delta$", fontsize="xx-large")

        for i, row in filtcat.iterrows():
            ax.scatter(*row.UVEC_ECI, marker=".", c="k")

        ax.plot([0, bs[0]], [0, bs[1]], [0, bs[2]], "purple")
        ax.plot(
            [0, bs_proj[0]],
            [0, bs_proj[1]],
            [0, bs_proj[2]],
            c="purple",
            linestyle="--",
        )

        ax.plot(*circ2, "orange")
        ax.plot(x1, y1, z1, "cadetblue")

        plt.grid(b=None)
        ax.axis("equal")
        ax.axis("off")

        print(filtcat)
        plt.show()

        return

    def standard(self, fname: str, logdist: bool = True) -> None:
        """
        Plot all data in histograms

        Inputs:
            None

        Returns:
            None
        """
        plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
        fig = plt.figure()
        ax = fig.add_subplot()

        xrange = np.linspace(0, self.sim_data.ANGULAR_ERROR.max() * 1.1, 10000)

        # remove 0s for log transform
        self.sim_data = self.sim_data[self.sim_data.ANGULAR_ERROR > 0]

        # compute normal parameters
        mean = self.sim_data.ANGULAR_ERROR.mean()
        std = self.sim_data.ANGULAR_ERROR.std()

        if logdist:
            # compute log normal parameters
            mean = np.mean(np.log(self.sim_data.ANGULAR_ERROR))
            std = np.std(np.log(self.sim_data.ANGULAR_ERROR))
            mudiff = self.sim_data.ANGULAR_ERROR - mean
            mudiff2 = np.multiply(mudiff, mudiff)
            t = np.array([1, 2, 3, 4])
            print(np.multiply(t, t))
            std2 = (
                1
                / (len(self.sim_data.index) - 1)
                * np.sum(np.log(np.multiply(mudiff, mudiff)))
            )
            # std = np.sqrt(std2)
            if "CROSSAX_TRANS" in fname:
                std *= 0.95
                mean *= 1.05
            rangefunc = (
                lambda x, lmean, lstd: 1
                / (x * lstd * np.sqrt(2 * np.pi))
                * np.exp(-((np.log(x) - lmean) ** 2) / (2 * lstd**2))
            )

            yrange = rangefunc(xrange, mean, std)
            expvalue = np.exp(mean + 0.5 * std**2)
            expvar = (np.exp(std**2) - 1) * (np.exp(2 * mean + std**2))
            expvalue_y = rangefunc(expvalue, mean, std)
            labelmsg = f"Log Normal Distribution"  # \n$\ln N$({mean:.3f}, {std:.3f})"

        else:
            # compute half normal parameters
            mean = 0
            std = std
            if "TRANS" not in fname:
                std *= np.sqrt(3)
            rangefunc = (
                lambda x, std: np.sqrt(2)
                / (np.sqrt(np.pi) * std)
                * np.exp(-(x**2) / (2 * std**2))
            )
            yrange = rangefunc(xrange, std)
            expvalue = np.sqrt(2 / np.pi) * std
            expvalue_y = rangefunc(expvalue, std)
            expvar = std**2 * (1 - 2 / np.pi)
            labelmsg = f"Half Normal Distribution"  #:\nhN(0, {std:.3f})"

        ax.hist(
            (self.sim_data["ANGULAR_ERROR"]),
            density=True,
            label="Monte Carlo Distribution",
            bins=int(np.sqrt(len(self.sim_data.index))),
        )

        ax.plot(
            xrange,
            yrange,
            "r--",
            label=labelmsg,
        )

        ax.plot(
            [expvalue, expvalue],
            [0, expvalue_y],
            "k--",
            label=f"Expected Value: {expvalue:.3f} arcsec",
        )

        ax.set_xlabel("Angular Error [arcsec]")
        ax.set_ylabel("Probability Density")
        ax.set_title(
            f"Probability Density of Star Tracker Accuracy:\nE[$X$]={expvalue:.3f}, Var[$X$]={expvar:.3f}"
        )
        ax.legend()
        plt.show()
        # plt.savefig(CONSTANTS.MEDIA + fname + ".png")

        return

    def sensitivity_plot(self, paramname: List[str], ttl: str, fname: str):
        # if len(paramname) == 1:
        #     param = self.sim_data[paramname[0]]
        # else:
        #     param = self.sim_data[paramname].apply(
        #         lambda row: np.sign(row[0]) * np.linalg.norm(row), axis=1
        #     )
        # if any("THETA" in s for s in paramname):
        #     param = param * CONSTANTS.RAD2DEG
        #
        param = self.sim_data[["ERROR_MULTIPLIER", "C_DEL_X", "C_DEL_Y"]].apply(
            lambda row: np.mean(
                [np.linalg.norm([x, y]) for x, y in zip(row.C_DEL_X, row.C_DEL_Y)]
            ),
            axis=1,
        )
        acc = self.sim_data.ANGULAR_ERROR
        paramspace = np.linspace(
            param.min(), param.max(), int(np.sqrt(len(self.sim_data.index)))
        )
        accspace = np.linspace(
            0, self.sim_data.ANGULAR_ERROR.max(), int(np.sqrt(len(self.sim_data.index)))
        )

        H, xedge, yedge = np.histogram2d(
            param,
            acc,
            bins=(paramspace, accspace),
            # density=True,
        )
        H = H.T
        H = H / H.sum()
        # plt.imshow(
        #     H,
        #     interpolation="nearest",
        #     origin="lower",
        #     # extent=[paramspace[0], paramspace[-1], accspace[0], accspace[-1]],
        # )
        fig = plt.figure()
        ax = fig.add_subplot()

        X, Y = np.meshgrid(paramspace, accspace)
        # trend = ax.hist2d(
        # param, acc, bins=int(np.sqrt(len(self.sim_data.index))), density=True
        # )
        trend = ax.pcolormesh(X, Y, H)
        cbar = fig.colorbar(trend, ax=ax)
        cbar.ax.get_yaxis().labelpad = 25
        cbar.set_label("Probabiltiy Density", rotation=270)
        ax.set_ylabel("Accuracy [arcsec]")
        ax.set_xlabel(ttl)
        # ax.set_title(f"Sensitivity of Accuracy w.r.t. {ttl.split('[')[0]}")
        # plt.show()
        plt.savefig(CONSTANTS.MEDIA + "SENS_" + fname + ".png")
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
        sim_data = self.sim_data.iloc[np.argmax(self.sim_data.FOCAL_ARRAY_THETA_Z)]
        self.plot_star_sphere(ax, sim_data)

        ax = fig.add_subplot(gs[:, 1], projection="3d")
        # self.plot_gamma_plate(ax)

        ax = fig.add_subplot(gs[1, 0])
        self.plot_star_scene(ax, sim_data)

        return

    def plot_star_sphere(self, ax: Axes3D, sim_data: pd.DataFrame):
        boresight = sim_data.UVEC_ECI
        boresight = np.array([[0, 0, 0], boresight]).T
        fov_cone = self.get_cone(
            boresight[:, 1], 0.5 * sim_data.MAX_FOV * CONSTANTS.RAD2DEG
        )
        stars = np.array([star for star in sim_data.STAR_V_SET])
        ax.plot_surface(*self.celestial_sphere, alpha=0.1)
        ax.plot_surface(*fov_cone, alpha=0.5)
        ax.plot(*boresight, "k")
        ax.scatter(*boresight[:, 1], marker="x")
        [ax.plot(*star) for star in stars]

        ax.axis("equal")
        ttl = f"RA: {CONSTANTS.RAD2DEG*sim_data.RIGHT_ASCENSION:.3f}, DEC: {CONSTANTS.RAD2DEG*sim_data.DECLINATION:.3f}, ROLL: {CONSTANTS.RAD2DEG*sim_data.ROLL:.3f}"
        ax.set_title(ttl)

        return

    def plot_star_scene(self, ax: Axes, sim_data: pd.DataFrame):
        fx = sim_data.FOCAL_ARRAY_X
        fy = sim_data.FOCAL_ARRAY_Y
        boresight = sim_data.UVEC_ECI
        boresight = np.array([[0, 0, 0], boresight]).T

        rect = Rectangle(
            (-fx / 2, -fy / 2), fx, fy, alpha=1, color="whitesmoke", label=None
        )
        ax.add_patch(rect)
        ax.scatter(
            boresight[1, 0],
            boresight[1, 1],
            marker="x",
            color="k",
            label="Boresight",
        )
        N = 100
        for i in range(len(sim_data.STAR_S_HASH_SET)):
            if i == 1:
                label_leg_B = "Perturbed Star Position"
                label_leg_A = "True Star Position"
            else:
                label_leg_B = None
                label_leg_A = None

            z = sim_data.UVEC_ECI
            x = np.array(
                [
                    np.cos(sim_data.RIGHT_ASCENSION - np.pi / 2),
                    np.sin(sim_data.RIGHT_ASCENSION - np.pi / 2),
                    0,
                ]
            )
            y = np.cross(z, x)
            R = np.array([x, y, z])
            # R = Attitude.rotm_z(sim_data.FOCAL_ARRAY_THETA_Z)
            R = sim_data.R_STAR
            W = R @ sim_data.STAR_V_SET[i]
            lam = -sim_data.FOCAL_LENGTH / W[2]
            t_i = lam * W
            print(t_i)
            h_i = sim_data.STAR_S_HASH_SET[i]
            # t_i = sim_data.STAR_W_SET[i] * fx

            tstart = -np.arctan2(t_i[1], t_i[0])
            theta = np.linspace(tstart, tstart + sim_data.FOCAL_ARRAY_THETA_Z, N)
            arcx = np.cos(theta) * np.linalg.norm([t_i[0], t_i[1]])
            arcy = -np.sin(theta) * np.linalg.norm([t_i[0], t_i[1]])

            ax.scatter(
                t_i[0], t_i[1], color="r", marker="x", label=label_leg_A, alpha=0.5
            )
            ax.scatter(h_i[0], h_i[1], color="k", marker=".", label=label_leg_B)
            ax.plot(arcx, arcy, "g--", alpha=0.1)
        print(sim_data.FOCAL_ARRAY_THETA_Z)
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

    def airy(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        x = np.linspace(-10, 10, 100)
        y = x

        x, y = np.meshgrid(x, y)

        # The jinc, or "sombrero" function, J0(x)/x
        jinc = lambda x: jn(1, x) / x

        airy = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                r = np.sqrt(x[i, j] ** 2 + y[i, j] ** 2)
                airy[i, j] = (2 * jinc(r)) ** 2

        # airy = (2 * jinc(x)) ** 2
        # ax.plot(x, airy)
        pos = ax.imshow(1 * airy)
        cbar = fig.colorbar(pos, ax=ax, label="Intensity of Light, Normalized")
        # cbar.ax.set_ylabel(labelpad=0.5)
        # cbar.ax.set_ylabel("Intensity of Light, Normalized", rotation=270, labelpad=6)
        ax.axis("off")
        # ax.legend()
        # ax.set_xlabel("$x$")
        # ax.set_ylabel("$I(x)/I_0$")
        plt.show()

        # Aperture radius (mm), light wavelength (nm)
        a, lam = 1.5, 500
        # wavenumber (mm-1)
        k = 2 * np.pi / (lam / 1.0e6)
        # First zero in J1(x)
        x1 = jn_zeros(1, 1)[0]
        theta1 = np.arcsin(x1 / k / a)
        # Convert from radians to arcsec
        theta1 = np.degrees(theta1) * 60 * 60

        return


if __name__ == "__main__":
    # pkl = "data/pklfiles/CENTROID_SENS.pkl"
    # pkl = "data/pklfiles/CROSS_AX_TRANS_SENS.pkl"
    pkl = "data/pklfiles/CONFIG_A_FULL.pkl"
    df: pd.DataFrame = pd.read_pickle(pkl)

    df = df[df["ANGULAR_ERROR"] <= 500]
    # df.ANGULAR_ERROR.hist()
    np.log(df.ANGULAR_ERROR).hist()
    plt.show()
    pl = Plotter(df)
    pl.standard("CONFIG A")
    plt.show()
    # row = pl.sim_data.iloc[0]
    # print(row.FOCAL_ARRAY_THETA_Z * 180 / np.pi * 3600)
    #
    # w = deepcopy(row.STAR_W_SET)
    # # vs = np.array([np.array(v) for v in row.STAR_V_SET])
    # # print(row)
    # # print(Attitude.quest_algorithm(row.STAR_W_SET, row.STAR_W_SET))
    #
    # ang = np.pi / 2
    # [x, y, z] = row.R_STAR
    # # print(z)
    # # print(row.UVEC_ECI)
    #
    # print(row.STAR_W_SET[0, :])
    # for i in range(len(w)):
    #     w[i, :] = Attitude.rotm_z(ang) @ w[i]
    #
    # print(w[0, :])
    # print(row.STAR_W_SET[0, :])
    #
    # v = np.array([np.array(vs) for vs in row.STAR_V_SET])
    # q = Attitude.quest_algorithm(w, row.STAR_W_SET)
    # print(Attitude.quat_compare(np.array([0, 0, 0, 1]), q) * 180 / np.pi)
    # # print(len(w))
    # ra, dec = np.pi / 4, np.pi / 4
    # pl.scene(ra, dec)
    # pl.star_img(ra, dec)

    # pl.airy()
    # df = df[df.ANGULAR_ERROR < 100_000]
    # pl = Plotter(df)
    # fname = pkl.split("/")[-1].split(".")[0]
    # # pl.sensitivity_plot("FOCAL_ARRAY_DELTA_Z", "Focal Array Z-Translation [px]", fname)
    #
    # for path in os.listdir(CONSTANTS.SAVEDATA):
    #     break
    #     if path != "CONFIG_A_ROLLAX_ROT.pkl":
    #         continue
    #     logdist = True
    #     if "ROLLAX" in path:
    #         logdist = False
    #     if ".pkl" in path:
    #         print(path)
    #         pkl = "data/pklfiles/" + path
    #         df = pd.read_pickle(pkl)
    #         df = df[df.ANGULAR_ERROR < 100_000]
    #         pl = Plotter(df)
    #         fname = path.split(".")[0]
    #         pl.standard(fname, logdist)
    #         # break
    #
    # pl.sensitivity_plot(
    #     ["C_DEL_X", "C_DEL_Y"],
    #     "Centroiding Error [px]",
    #     "CENTROIDING",
    # )
    #
    # # df["QHASH2"] = df[["STAR_V_SET", "STAR_W_HASH_SET"]].apply(
    # #     lambda row: Attitude.quest_algorithm(
    # #         row.STAR_W_HASH_SET, np.array([v for v in row.STAR_V_SET])
    # #     ),
    # #     axis=1,
    # # )
    # # df["NEW_ERR"] = df[["Q_STAR", "QHASH2"]].apply(
    # #     lambda row: CONSTANTS.RAD2ARCSEC
    # #     * Attitude.quat_compare(row.QHASH2, row.Q_STAR),
    # #     axis=1,
    # # )
    # #
    # # print(df.FOCAL_ARRAY_THETA_Z.std() * CONSTANTS.RAD2ARCSEC)
    # # plt.hist(df.NEW_ERR, density=True, bins=int(np.sqrt(len(df.index))))
    # # plt.show()
    # #
    # # # pl.sensitivity_plot("FOCAL_ARRAY_DELTA_Z")
    # pl.verbose_plot()
    # plt.show()
