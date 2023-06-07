"""
Functions to compute and visualize the KDE

Implemented for the 2D reaction coordinates plot.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde


def kde(x, y, bw_method, n_mesh, extrema=None) -> tuple:
    """
    Perform a 2D Kernel Density Estimation from x and y data.
    This function is an interface for scipy.stats.gaussian_kde
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    More info are described in its documentation and in the
    stdout this function prints.

    Parameters
    ----------
    x : array_like
        1D array/list containing data for the x direction
    y : array_like
        1D array/list containing data for the y direction
    bw_method : str or float
        If `str`, select a method implemented in gaussian_kde
        to compute a bandwidth parameter. If `float`, the
        `covariant_factor` is the one specified by the user
    n_mesh : int
        Number of points along one direction in which evaluate
        the KDE
    extrema : Dict, optional
        Extrema of the KDE evaluation, by default None

    Returns
    -------
    tuple
        X,Y : array_like
            Mesh grid of the evaluated KDE (output of `np.meshgrid`)
        Z : array_like
            2D array with the KDE evaluated in the X,Y points
        Kernel : gaussian_kde object
            For more details see teh gaussian_kde doc
    """
    data = np.vstack((x, y))
    kernel = gaussian_kde(data, bw_method=bw_method)

    # factor evaluated by gaussian_kde describing the
    # quantity by which the std is multiply to have the sigma
    # for the gaussians
    f = kernel.covariance_factor()
    print(
        f"""
    Covariance factor = {f}; bw = cf*std:
    For x: {f * x.std()}
    For y: {f * y.std()}\n
    """
    )

    if extrema is None:
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
    else:
        xmin = extrema["xmin"]
        xmax = extrema["xmax"]
        ymin = extrema["ymin"]
        ymax = extrema["ymax"]
    X, Y = np.meshgrid(np.linspace(xmin, xmax, n_mesh), np.linspace(ymin, ymax, n_mesh))
    xy = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel.pdf(xy), X.shape)

    return (X, Y, Z, kernel)


def contour_custom(
    X, Y, Z, lims, ax, fig, plot_params
) -> matplotlib.contour.QuadContourSet:
    """
    _summary_

    Parameters
    ----------
    X,Y : array_like
            Mesh grid of the evaluated function (output of `np.meshgrid`)
        Z : array_like
            2D array with the function evaluated in the X,Y points
    lims : tuple(tuple(float, float), tuple(float, float))
        Limits in which draw the contour plot (first x then y)
    ax : Axes
        Axes where to draw
    fig : Figure
        Figure object where the plot is drawn
    plot_params : Dict
        Parameters to draw the contour. For finer details see the code

    Returns
    -------
    Contourf
        Output of plt.contourf
    """
    level_difference = plot_params["levels_difference"]
    maximum_level_2show = plot_params["last_level"]
    cmap = plot_params["cmap"]

    levels = np.linspace(
        0, maximum_level_2show, round(maximum_level_2show / level_difference) + 1
    )
    print(f"Levels lower bounds:\n{levels}")

    if isinstance(cmap, matplotlib.colors.ListedColormap):
        print("Using CMAP")
        CS = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    elif isinstance(cmap, str):
        print(f"Loading colors from {cmap}")
        colors_list = np.loadtxt(cmap, dtype=float, comments="#")
        assert len(colors_list) >= len(levels)
        CS = ax.contourf(X, Y, Z, levels=levels, colors=colors_list)
    else:
        raise RuntimeError("Invalid CMAP")

    # colorbar setting
    if plot_params["cbar"]:
        cbaxes = inset_axes(
            ax,
            height="3%",
            width="35%",
            loc=plot_params["cbar_position"],
            borderpad=1.3,
        )
        c = fig.colorbar(
            CS,
            ax=ax,
            cax=cbaxes,
            orientation="horizontal",
            ticklocation=plot_params["cbar_ticks_position"],
        )
        c.set_ticks(
            ticks=(levels[:: plot_params["cbar_ticks_every_jump"]]),
            labels=[
                f"{x:.1f}" for x in levels[:: plot_params["cbar_ticks_every_jump"]]
            ],
        )
        c.ax.tick_params(labelsize=plot_params["cbar_labelsize"], pad=2)
        c.set_label(
            plot_params["cbar_label"],
            labelpad=plot_params["cbar_labelpad"],
            fontsize=plot_params["cbar_fontsize"],
        )

    # ax customization
    ax.yaxis.set_major_locator(plt.FixedLocator(plot_params["y_major_locator"]))
    ax.yaxis.set_major_formatter(plt.FixedFormatter(plot_params["y_major_formatter"]))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(plot_params["y_minor_locator"]))
    ax.xaxis.set_major_locator(plt.FixedLocator(plot_params["x_major_locator"]))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(plot_params["x_major_formatter"]))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(plot_params["x_minor_locator"]))

    ax.grid(alpha=0.4, which="both", axis="y")
    ax.grid(alpha=0.4, which="major", axis="x")

    ax.set_ylabel(plot_params["ylabel"])
    ax.set_xlabel(plot_params["xlabel"])

    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])

    return CS
