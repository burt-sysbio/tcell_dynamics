import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
sns.set(context = "poster", style = "ticks", rc = {"lines.linewidth": 5})

pscan = pd.read_csv("../output/output_fig2B_data_estimates_linear.csv")

pscan = pscan[pscan.readout != "Decay"]

# divide by alpha to get mean instead of rate
pscan.p_val = pscan.p_val.apply(lambda x : x/7.)

pscan["norm2"] = pscan.groupby(["readout", "name"])["read_val"].transform(lambda x: np.log2(x/x.min()))
# rename some readouts
pscan.loc[pscan["readout"] == "Area", "readout"] = "Response Size"
pscan.loc[pscan["readout"] == "Peak", "readout"] = "Peak Height"
pscan.loc[pscan["readout"] == "Peaktime", "readout"] = "Peak Time"

# remove null model
pscan = pscan.loc[pscan["name"] != "Null"]

g = sns.relplot(data = pscan, x = "p_val", y = "norm2", col = "name", hue = "readout",
                kind = "line", facet_kws = {"despine" : False})
g.set(ylim = (-5,5), xlabel = "divisions per day",
      ylabel = "effect size")

g.set_titles("{col_name}")
for ax in g.axes:
    for a in ax:
        #a.set_title("")
        #a.xaxis.set_major_locator(loc_major)
        #a.xaxis.set_minor_locator(loc_minor)
        #a.axhline(y = 0, linewidth = 2., ls = "--", color = "k", zorder = 1000)
        a.axvline(x = 15.2/7, linewidth = 2., ls = "--", color = "k", zorder = 1000)

plt.show()
g.savefig("../figures/fig2B_linear.svg")
g.savefig("../figures/fig2B_linear.pdf")
