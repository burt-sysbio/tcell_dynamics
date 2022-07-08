import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import final_figures.code_final_figures.models as model
from final_figures.code_final_figures.params_fig2 import d
from final_figures.code_final_figures.exp_final_figures import Simulation

d = dict(d)
sns.set(context = "poster", style = "ticks")
loaddir = "../output/"

df = pd.read_csv(loaddir + "repeated_stimulation_IL2_Timer.csv")
df2 = pd.read_csv(loaddir + "repeated_stimulation_IL2.csv")

df3 = pd.concat([df,df2])
nsim1 = len(df["ID"].drop_duplicates())
nsim2 = len(df2["ID"].drop_duplicates())

mu_arr = df["mu"].drop_duplicates().sort_values()
mu1 = mu_arr.iloc[0]
mu2 = mu_arr.iloc[20]
mu3 = mu_arr.iloc[29]

df_red = df.loc[df["mu"] == mu2]
df2_red = df2.loc[df2["mu"] == mu2]
colors = ["k", "purple"]

# plot example for timeril2 and il2 restimulation (1 stimulation frequency)
fig, ax = plt.subplots(figsize= (6.,5))
for mydf, nsim, c in zip([df2_red, df_red], [nsim2, nsim1], colors):
    sns.lineplot(data = mydf, x = "time", y = "value",
                 hue = "ID", palette = [c]*nsim, alpha = 0.2, lw = 2, ax = ax, legend = False)

ax.set_xlim([0,50])
ax.set_xticks([0,10,20,30,40,50])
ax.set_yscale("log")
ax.set_xlabel("time (d)")
ax.set_ylabel("cells")
ax.set_ylim([1,None])
plt.tight_layout()
plt.show()
fig.savefig("../final_figures_new/fig2C.svg")
fig.savefig("../final_figures_new/fig2C.pdf")

#
cells = df3.loc[df3["mu"].isin([mu1,mu2,mu3])]

g = sns.relplot(data = cells, x = "time", y = "value", col = "mu", kind = "line",
                row = "name", hue = "ID", alpha = 0.1, lw = 2, palette=["k"]*nsim, legend = False)
g.set_titles("")
sns.despine(top = False, right = False)
g.set(yscale = "log", ylim = [1, None], xlabel = "time (d)",
      ylabel = "cells", xlim = [0,80])
plt.show()
g.savefig("../final_figures_new/supp_timecourse_restim.svg")
#
#
df_grouped = df3.groupby(["mu", "ID", "name"])["value"].agg("max").reset_index()
# plot scan where mu is varied

g = sns.relplot(data = df_grouped, x = "mu", y = "value", kind = "line", hue = "name",
                palette = colors, aspect = 1.2)
g.set(yscale = "log", xscale = "log", xlabel = "restimulation events / day ",
      ylabel = "Peak Height", xlim = [0.01,1.0])
sns.despine(top = False, right = False)
plt.show()

g.savefig("../final_figures_new/fig2D.svg")
g.savefig("../final_figures_new/fig2D.pdf")

# =============================================================================
# make time course
# =============================================================================
start_times = [(0,50)]

def vir_model_const(t):
    return 0

names = ["IL2", "Timer", "IL2_Timer"]
mymodels = [model.il2_prolif, model.timer_prolif, model.timer_il2_prolif]
mylist_cntrl = []
mylist_prolif = []
prolif_arr = [15,28.7]
mylists = [mylist_cntrl, mylist_prolif]

for name, mymodel in zip(names, mymodels):
    sim = Simulation(name=name, mode=mymodel, parameters=d,
                     start_times=start_times,
                      vir_model=vir_model_const)

    for mylist, val in zip(mylists, prolif_arr):
        sim.parameters["beta_p"] = val
        sim.compute_cellstates()
        df = sim.get_readouts()
        df["param_value"] = val
        mylist.append(df)

df_cntrl = pd.concat(mylist_cntrl)
df_prolif = pd.concat(mylist_prolif)

df_prolif["val_norm"] = np.log2(df_prolif["read_val"] / df_cntrl["read_val"])
df_prolif = df_prolif.loc[df_prolif["readout"] == "Peak", ["val_norm", "name"]]
df_prolif["stim"] = "prolif"

# aggregate output from restimulation
out = df_grouped.groupby(["mu", "name"])["value"].agg(["mean"]).reset_index()
out["val_norm"] = out.groupby(["name"])["mean"].transform(lambda x : np.log2(x/x.min()))

# find appropriate mu that matches proliferation
out = out.loc[out["mu"].isin([mu1, mu_arr.iloc[15]])]

out = out.loc[out["val_norm"] != 0, ["val_norm", "name"]]
# add dummy column for Timer which is not affected by restimulation
new_row = {'val_norm': 0, "name" : "Timer"}
out = out.append(new_row, ignore_index = True)
out["stim"] = "restim"

df = pd.concat([df_prolif, out])

g = sns.catplot(data = df, x = "name", y = "val_norm", hue = "stim", kind = "bar", palette = ["k", "grey"],
                aspect = 1.1)
g.set(xlabel = "", ylabel = "FC Peak Height")
g.set_xticklabels(rotation = 90)
sns.despine(top = False, right = False)
plt.show()

g.savefig("../final_figures_new/fig2E.svg")