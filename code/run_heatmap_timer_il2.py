#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:08:29 2020

@author: burt

plot heatmap antigen versus proliferation for il2 and il2+timer models
"""
import sys
sys.path.append("../../")
from final_figures.code_final_figures.exp_final_figures import Simulation

import final_figures.code_final_figures.models as model
from final_figures.code_final_figures.params_fig2 import d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = "poster", style = "ticks")


# d_data = {
#     "b": 0,
#     "initial_cells": 1.0,
#     "alpha": 10,
#     "beta": 10 / (35/24), # first division around 35 hours
#     "alpha_p": 8, # I think in original fit its 7 but I need alpha%2 for intermediate cells
#     "beta_p": 8 / (11/24), # alpha / mu
#     "lifetime_eff": 1/0.24, # death rate is 0.24
#     "d_prec": 0,
#     "d_naive": 0,
#     "n_div": 2,
#     "rate_il2": 5e9,
#     "rate_C": 15.0,
#     "deg_myc": 0.1,
#     "K_IL2": 0.1,
#     "K_C": 1.0,
#     "K_myc": 0.1,
#     "hill": 3,
#     "crit": False,
#     "t0": None,
#     "c_il2_ex": 0,
#     "up_il2": 1,
#     "deg_il2" : 0,
# }

d = dict(d)
d["initial_cells"] = 100
d["up_il2"] = 1 * 3600 * 24
# =============================================================================
# set up models
# =============================================================================
time = [(0,50)]

sim1 = Simulation(name="Timer_IL2", mode=model.timer_il2_prolif, parameters=d, start_times=time)

sim2 = Simulation(name="Timer", mode=model.timer_prolif, parameters=d, start_times=time)

sim3 = Simulation(name="IL2", mode=model.il2_prolif, parameters=d, start_times=time)

sims = [sim1, sim2, sim3]


fig, axes = plt.subplots(1, 3, figsize=(20, 4))
for ax, sim in zip(axes, sims):
    sim.compute_cellstates()
    df = sim.cells.loc[sim.cells["species"] == "CD4_all"]
    sns.lineplot(x="time", y="value", color="crimson",
                 data=df, ax=ax)
    ax.set_ylim(1, None)
    ax.axhline(1e3, c = "k", ls = "--")
    ax.axvline(8, c="k", ls="--")
    ax.set_ylabel("")
        #ax.set_xticks(np.arange(0,time[-1], 3))
    ax.set_yscale("log")
    ax.set_title(sim.name)
axes[0].set_ylabel("cells")
plt.show()

#=============================================================================
#general parameters
#=============================================================================
res = 50
savedir = "../output/heatmaps/"
sname = "heatmap_Timer_IL2_"

# =============================================================================
# vary IL2 secretion rate and myc
# =============================================================================
name1 = "up_il2"
name2 = "deg_myc"

il2_default = d["up_il2"]
myc_default = d["deg_myc"]
arr1 = np.geomspace(il2_default*0.01,il2_default*100, res)
arr2 = np.geomspace(myc_default*0.1,myc_default*10, res)

print("starting 2d scan")
df_heatmap = sim1.get_heatmap(arr1, arr2, name1, name2)
df_heatmap.to_csv(savedir + sname + name1 + "_data_estimate.csv", index = False)


