#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:14:25 2020

@author: burt
"""
import sys
sys.path.append("../../")
from exp import Simulation
import models as model
from params_fig2 import d
import numpy as np
import matplotlib as mpl
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("paper_theme_python.mplstyle")
sns.set_palette("deep")

from scipy.interpolate import interp1d

d = dict(d)
d["initial_cells"] = 1
d["deg_restim"] = 0.1
d["deg_myc"] = 0.2 # make it a slow timer so that timer only kicks in once IL2 mechanism fails
#d["deg_myc"] = 1.0 # this is the timer for restimulation

def vir_model_const(t):
    return 0
# for carrying capacacity I need a long long timecourse
# because the peak comes so late

# for timer and IL2 timecourse needs to be a bit longer
# for the cells to return to 0 at the end because they have a high peak


res = 50
arr = np.geomspace(0.1,10,res)


start_times = [(0, 100)]
t_eval = np.arange(0,80,0.1)
max_step = 0.1

sim_il2 = Simulation(name = "IL2", mode = model.restim, parameters = d, start_times = start_times,
                     vir_model = vir_model_const)

sim_timer_il2 = Simulation(name = "IL2_Timer", mode = model.restim_timer_il2, parameters = d, start_times = start_times,
                     vir_model = vir_model_const)

pname = "il2_stimulation"
#out1 = sim_il2.vary_param(pname = pname, arr = arr, normtype= "middle", t_eval = np.arange(0,80,0.1), max_step = max_step)

cell_list = []
mol_list = []
for val in arr:
    sim_il2.parameters[pname] = val
    sim_il2.compute_cellstates(t_eval = t_eval)

    cells = sim_il2.cells
    mols = sim_il2.mols

    cells["param_value"] = val
    cells["param_name"] = pname
    cell_list.append(cells)

    mols["param_value"] = val
    mols["param_name"] = pname
    mol_list.append(mols)

cells = pd.concat(cell_list).reset_index(drop = True)
mols = pd.concat(mol_list).reset_index(drop = True)


cells = cells.loc[cells.species == "CD4_all"]


cmap = "rocket_r"
g = sns.relplot(data = cells, x = "time", y = "value", hue = "param_value", kind = "line",
                height = 1.8, legend = False, palette = cmap, hue_norm = mpl.colors.LogNorm())
g.set(yscale ="log", ylim = [1, None], ylabel = "cells", xlabel = "time (h)", xticks = [0,20,40,60,80],
      xlim = [0,80])
sns.despine(top = False, right = False)


sm = plt.cm.ScalarMappable(cmap=cmap, norm= mpl.colors.LogNorm(vmin=arr.min(), vmax=arr.max()))
plt.colorbar(sm)
plt.show()

g.savefig("../figures/supplements/cells_il2_perturbation.pdf")
g.savefig("../figures/supplements/cells_il2_perturbation.svg")

g = sns.relplot(data = mols, x = "time", y = "value", hue = "param_value",
                col = "species", facet_kws = {"sharey" : False}, kind = "line", height = 1.8,
                palette = cmap, hue_norm = mpl.colors.LogNorm(), aspect = 0.9)
g.set(xlabel = "time (h)", ylabel = "conc.", xticks = [0,20,40,60,80], xlim = [0,80])
sns.despine(top = False, right = False)
plt.show()

g.savefig("../figures/supplements/molecules_il2_perturbation.pdf")
g.savefig("../figures/supplements/molecules_il2_perturbation.svg")
