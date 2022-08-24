
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pl import plot_heatmap
plt.style.use("paper_theme_python.mplstyle")
############################################
############################################
############################################
# proliferation vs antigen
cmap = sns.color_palette("rocket_r", as_cmap=True)


# plot params
xlabel = "IL2 uptake / secretion"
ylabel = "Timer degradation"
title = "Timer_IL2"
vmin = 0
vmax = 10
myvalue = "val_norm"

pname = "up_il2"

def load_heatmap(pname, loaddir = "../output/heatmaps/"):
    df = pd.read_csv(loaddir + "heatmap_Timer_IL2_" + pname + "_data_estimate.csv")
    df["val_norm"] = df.groupby(["readout"])["value"].transform(lambda x: np.log2(x / x.min()))
    return df


#for pname in pnames:
df = load_heatmap(pname)

il2_secretion = 3600*24*100
df["param_val1"] = df["param_val1"] / (il2_secretion)

# rename area to Response size
df.loc[df["readout"] == "Area", "readout"] = "Response Size"

contour_levels = [[1e3, 1e4], [7,8,9], [1e3,1e4]]
readouts = df["readout"].drop_duplicates().values
for readout, levels in zip(readouts, contour_levels):
    if readout == "Peak Time":
        vmax = 2
    elif readout != "Peak Time":
        vmax = 15

    fig, z = plot_heatmap(df, value_col=myvalue, readout=readout, log_color=False,
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       log_axes=True, xlabel=xlabel, ylabel=ylabel, title=title, contour_levels = None,
                          figsize= (2.2,1.8))
    plt.show()

    savename = "heatmap_timer_il2_" + readout
    savedir = "../figures/supplements/"
    fig.savefig(savedir + savename + ".svg")