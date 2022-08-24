import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib


def plot_timecourse(df, cells, hue = None, col = None, ylim = None,
                    row = None, yscale = "log", style = None, palette = None, *kwargs):
    """
    take either cells or molecules from run sim object
    """
    # provide cells to plot in cells array
    df = df.loc[df.cell.isin(cells)]
    # only focus on effector cells, not chronic and total cells
    g = sns.relplot(data=df, x="time", palette = palette, hue = hue, col=col, row = row,
                    y="value", style = style, kind="line", *kwargs)

    g.set(ylim=ylim, ylabel = "cells")
    if yscale == "log":
        g.set(yscale = "log")
    g.set_titles("{col_name}")
    return g


def plot_pscan(df, cells, readouts = ["Area", "Peak", "Peaktime", "Decay"], xscale = "log", yscale = None,
               hue = None, ylim = None, xlim = None,
               value_col = "val_norm", col = "readout", row = None, palette = None, xlabel = None,
               ylabel = None, **kwargs):
    """
    take df generated through pscan function
    """
    if (len(cells) > 1) & (hue is None):
        hue = "cell"
    df = df.loc[df.cell.isin(cells) & (df.readout.isin(readouts))]
    g = sns.relplot(data = df, x = "param_value", y = value_col, col = col, row = row,
                    hue = hue, palette= palette, **kwargs)
    if xscale is not None:
        g.set(xscale = xscale)

    if xlabel is None:
        g.set(xlabel = df.param.iloc[0])
    else:
        g.set(xlabel = xlabel)

    if ylabel is not None:
        g.set(ylabel = ylabel)
    else:
        g.set(ylabel = "effect size")

    if yscale is not None:
        g.set(yscale = "log")

    if ylim is not None:
        g.set(ylim = ylim)

    if xlim is not None:
        g.set(xlim = xlim)
    else:
        g.set(xlim = (df.param_value.min(), df.param_value.max()))

    g.set_titles("{col_name}")
    return g


def plot_heatmap(df, value_col, readout, log_color, xlabel = None, ylabel = None,
                 vmin=None, vmax=None, cmap="Reds", log_axes=True, cbar_label = None,
                 title = None, contour_levels = None, figsize = (4, 3)):
    """
    NOTE that I changed this function to apply to data models works with pscan2d from proc branch rtm
    df needs to have a column named readout and a column named value or similar
    take df generated from 2dscan and plot single heatmap for a given readout
    note that only effector cells are plotted
    value_col: could be either val norm or value as string
    log_color: color representation within the heatmap as log scale, use if input arr was log scaled
    or if variation across multiple scales is expected
    """
    # process data (df contains all readouts and all cells
    df = df.loc[df["readout"] == readout,:]
    arr1 = df["param_val1"].drop_duplicates()
    arr2 = df["param_val2"].drop_duplicates()
    assert (len(arr1) == len(arr2))

    # arr1 and arr2 extrema are bounds, and z should be inside those bounds
    z_arr = df[value_col].values
    z = z_arr.reshape((len(arr1), len(arr2)))
    z = z.T
    q = z
    z = z[:-1, :-1]
    # transform because reshape somehow transposes this
    #z=z.T
    # check if color representation should be log scale
    sm, norm = get_colorscale(log_color, cmap, vmin, vmax)

    # plot data
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(arr1, arr2, z, norm = norm, cmap=cmap, rasterized = True)
    if contour_levels is not None:
        ax.contour(arr1, arr2, q, norm=norm, levels = contour_levels, colors = ["white"])
    # tick reformatting
    loc_major = ticker.LogLocator(base=10.0, numticks=100)
    loc_minor = ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=12)

    # adjust scales
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        #ax.xaxis.set_major_locator(loc_major)
        #ax.xaxis.set_minor_locator(loc_minor)

    if xlabel is None:
        xlabel = ax.set_xlabel(df.pname1.iloc[0])
    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = ax.set_ylabel(df.pname2.iloc[0])
    ax.set_ylabel(ylabel)

    cbar = plt.colorbar(sm, ax=ax)
    if cbar_label is None:
        cbar_label = readout
    cbar.set_label(cbar_label)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    return fig, z


def get_colorscale(hue_log, cmap, vmin = None, vmax = None):
    if hue_log:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # make mappable for colorbar
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return sm, norm


def plot_timecourses(df,
                     hue_log,
                     pl_cbar,
                     cbar_label = None,
                     cells = None,
                     scale=None,
                     xlabel = "time (d)",
                     ylabel = "n cells",
                     xlim=(None, None),
                     ylim=(None, None),
                     cmap="Greys",
                     cbar_scale=1.,
                     arange_ticks=None,
                     hue = "param_value",
                     vmin = None,
                     vmax = None,
                     sharey = False,
                     leg_title = None,
                     show_titles = True,
                     **kwargs):
    """
    pl_cbar: bool - draw color bar? Note that this only works with Greys and Blues cmap atm
    hue_log : bool - whether color mapping should be in log space use if df arr is log spaced
    cells : list - which cell types?
    scale : None or "log"
    ticks : None or list of ticks
    leg_title - str - legend title

    scale - None or "log" - yscale
    arange_ticks : if True, tries to find smart ticks, if None, default, if list, sets ticks to list
    """
    # only plot cells in cells arr
    if cells is not None:
            df = df.loc[df.cell.isin(cells)]

    # parameter for scaling of color palette in sns plot
    arr = df.param_value.drop_duplicates()

    if (cmap == "Greys") or (cmap == "Blues") or (cmap == "Greys_r"):
        sm, hue_norm = get_colorscale(hue_log, cmap, vmin, vmax)
    else:
        sm, hue_norm = None, None

    # hue takes the model name, so this should be a scalar variable
    # can be generated by change_param function
    g = sns.relplot(x="time", y="value", kind="line", data=df, hue=hue,
                    hue_norm=hue_norm, palette=cmap,
                    facet_kws={"despine": False, "sharey" : sharey}, **kwargs)

    g.set(xlim=xlim, ylim=ylim, xlabel = xlabel, ylabel = ylabel)

    if scale is not None:
        assert scale == "log"
        g.set(yscale = scale)

    if show_titles:
        g.set_titles("{col_name}")
    else:
        g.set_titles("")
    if leg_title is not None:
        g._legend.set_title(leg_title)

    # if ticks are true take the upper lower and middle part as ticks
    # for colorbar
    if pl_cbar:
        assert sm is not None
        if arange_ticks:
            if hue_log:
                ticks = np.geomspace(np.min(arr), np.max(arr), 3)
            else:
                ticks = np.linspace(np.min(arr), np.max(arr), 3)

            cbar = g.fig.colorbar(sm, ticks=ticks)
            cbar.ax.set_yticklabels(np.round(cbar_scale * ticks, 2))
        else:
            cbar = g.fig.colorbar(sm, ticks=arange_ticks)

        # add colorbar
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        else:
            pname = df.pname.iloc[0]
            cbar.set_label(pname)
        cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    return g
