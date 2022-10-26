#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:14:25 2020

@author: burt
"""

from exp_repeated_stimulation import Simulation, SimList, make_sim_list
import models_repeated_stimulation as model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("paper_theme_python.mplstyle")
sns.set_palette("deep")
from scipy.constants import N_A
d = {
    "b": 0,
    "initial_cells": 1.0,
    "alpha": 4,
    "beta": 2.,
    "alpha_p": 2,
    "beta_p": 4.5,
    "d_prec": 0,
    "d_naive": 0,
    "d_eff" : 0.24,
    "n_div": 2,
    "rate_il2": 300 * 3600 * 24,
    "rate_il2_restim": 300 * 3600 * 24,
    "deg_myc": 0.37,
    "deg_il2_restim" : 6.25,
    "K_il2_cons": 7.5*N_A*20e-6*10e-12,
    "K_il2" : 5,
    "K_myc": 0.1,
    "hill": 3,
    "c_il2_ex": 0,
    "up_il2": 1 * 3600 * 24,
    "deg_il2" : 0
}


# for carrying capacacity I need a long long timecourse
# because the peak comes so late

# for timer and IL2 timecourse needs to be a bit longer
# for the cells to return to 0 at the end because they have a high peak
def gen_start_times(dur, res = 1000, end_runtime = 100, end_stimulation = 20):
    start_times = [(i * dur, (i + 1.0) * dur) for i in range(res)]
    start_times = [(x, y) for x, y in start_times if y <= end_stimulation]
    assert len(start_times) > 0
    start_times.append((start_times[-1][1], end_runtime))

    return start_times

def run_stimulations(dur_arr, res = 1000, end_runtime = 100, end_stimulation = 20):
    """

    """
    cells = []
    mols = []
    for dur in dur_arr:
        start_times = [(i*dur, (i+1.0)*dur) for i in range(res)]
        # only keep start_times until the end_stimulation value
        start_times = [(x, y) for x, y in start_times if y <= end_stimulation]
        assert len(start_times)>0
        start_times.append((start_times[-1][1], end_runtime))

        sim1 = Simulation(name="IL2", mode=model.il2_menten_prolif, parameters=d,
                          core=model.diff_effector, start_times = start_times)

        sim2 = Simulation(name="Timer", mode=model.timer_menten_prolif, parameters=d,
                          core=model.diff_effector, start_times = start_times)

        sim3 = Simulation(name="Mixed", mode=model.timer_il2_menten, parameters=d,
                          core=model.diff_effector, start_times = start_times)


        #check that model output is similar by adjusting model specific parameters then plot
        df1 = sim1.run_timecourse()
        df2 = sim2.run_timecourse()
        df3 = sim3.run_timecourse()


        df = pd.concat([df1, df2, df3]).reset_index(drop = True)
        df_mols = pd.concat([sim1.molecules_tidy, sim2.molecules_tidy, sim3.molecules_tidy]).reset_index(drop=True)
        df["dur"] = dur
        df_mols["dur"] = dur
        cells.append(df)
        mols.append(df_mols)
    cells = pd.concat(cells).reset_index()
    mols = pd.concat(mols).reset_index()

    return cells, mols


start_times = [(0,5),(5,10),(10,50),(50,51),(51,52),(52,53),(54,55), (55,56), (56,57)]
#start_times = [(0,50),(50,60)]
sim1 = Simulation(name="IL2", mode=model.il2_menten_prolif, parameters=d,
                  core=model.diff_effector, start_times=start_times)

sim2 = Simulation(name="Timer", mode=model.timer_menten_prolif, parameters=d,
                  core=model.diff_effector, start_times=start_times)

sim3 = Simulation(name="Mixed", mode=model.timer_il2_menten, parameters=d,
                  core=model.diff_effector, start_times=start_times)


df_list = []
df_list2 = []
sim_list = [sim1]
n= 1
for i in range(n):
    # check that model output is similar by adjusting model specific parameters then plot
    for sim in sim_list:
        df = sim.run_timecourse()
        df_list.append(df)
        df_list2.append(sim.molecules_tidy)

df = pd.concat(df_list).reset_index(drop=True)
df_mols = pd.concat(df_list2).reset_index(drop=True)

g = sns.relplot(data = df, x = "time",y = "cells", kind = "line", hue = "name", height = 1.8)
g.set(xlim= (0,None), yscale = "log", ylim = [1,None], xlabel = "time (d)")
sns.despine(top=False, right=False)
plt.show()

g.savefig("ensemble_stimulation_timecourse_2.pdf")
g.savefig("ensemble_stimulation_timecourse_2.svg")


g = sns.relplot(data = df_mols, x = "time",y = "value", col = "variable", facet_kws = {"sharey" : False},
                kind = "line", hue = "name", height = 1.8)
g.set(xlabel = "time (d)", yscale = "log", ylim = [0.1,10])
sns.despine(top=False, right=False)
plt.show()

dur_arr = [0.5,1,2,4,10]
#
# cells, mols = run_stimulations(dur_arr)
# g = sns.relplot(data = cells, x = "time",y = "cells", kind = "line", col = "name", hue = "dur", height = 2)
# g.set(xlim= (0,50), yscale = "log", ylim = [1,None])
# plt.show()
#
# g = sns.relplot(data = mols, x = "time",y = "value", row = "variable", facet_kws = {"sharey" : False},
#                 kind = "line", hue = "dur", col = "name", height = 2)
# plt.show()
#
# #
# out = cells.groupby(["dur", "name"])["cells"].agg("max").reset_index()
# out["freq"] = 1 / out["dur"]
# g = sns.relplot(data = out, x = "freq", y = "cells", hue = "name", height = 2)
# plt.show()