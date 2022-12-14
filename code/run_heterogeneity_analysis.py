#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:14:25 2020

@author: burt
"""

from exp import Simulation, SimList
import models as model
from params_fig2 import d
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
d = dict(d)
# =============================================================================
# make time course
# =============================================================================
import os

print(os.getcwd())

# for timer and IL2 timecourse needs to be a bit longer
# for the cells to return to 0 at the end because they have a high peak
time_il2 = [(0, 100)]
time_timer = [(0, 100)]

t_eval = np.arange(0,80,0.1)

def vir_model_const(t):
    return 0


sim_il2 = Simulation(name="IL2", mode=model.il2_prolif, parameters=d, start_times=time_il2,
                  vir_model=vir_model_const)

sim_timer = Simulation(name="Timer", mode=model.timer_prolif, parameters=d, start_times=time_timer,
                  vir_model=vir_model_const)

# =============================================================================
# make parameter scan
# =============================================================================
# check if dynamics are similar for default params
sim_timer.parameters["deg_myc"] = 0.395

cells_il2 = sim_il2.compute_cellstates(t_eval = t_eval, max_step = np.inf)
cells_timer = sim_timer.compute_cellstates(t_eval = t_eval, max_step = np.inf)


cells_il2 = cells_il2.loc[cells_il2.species == "CD4_all"]
cells_timer = cells_timer.loc[cells_timer.species == "CD4_all"]

cells = pd.concat([cells_il2, cells_timer]).reset_index(drop = True)

g = sns.relplot(data = cells, x = "time",y = "value", hue = "name", kind = "line")
plt.show()

def run_heterogeneity(sim, CV : float, res : int, pnames : list, sname : str, **kwargs):
    cell_list = []

    for i in range(res):
        sim.set_params_lognorm(pnames, CV)

        cells = sim.compute_cellstates(**kwargs)
        cells["run_ID"] = i
        cell_list.append(cells)
        sim.reset_params()


    df = pd.concat(cell_list).reset_index()
    df["CV"] = CV
    df.to_csv("../output/heterogeneity/heterogeneity_cv_" + str(CV) + "_" + sim.name + "_" + sname + ".csv", index = False)


pnames = ["beta_p"]

pnames_large = ["beta_p", "deg_myc", "beta_prec", "up_il2", "rate_il2_prec"]

sname = "prolif"
sname_large = "allparams"
res = 50

CV_list = [0.01,0.05,0.1,0.5,1.0]
for CV in CV_list:
    # set max step otherwise dataframes get too large
    run_heterogeneity(sim_il2, CV, res, pnames, sname, t_eval=t_eval, max_step = np.inf)
    run_heterogeneity(sim_timer, CV, res, pnames, sname, t_eval=t_eval, max_step = np.inf)

    #run_heterogeneity(sim_il2, CV, res, pnames_large, sname_large, t_eval=t_eval, max_step = np.inf)
    #run_heterogeneity(sim_timer, CV, res, pnames_large, sname_large, t_eval=t_eval, max_step = np.inf)

