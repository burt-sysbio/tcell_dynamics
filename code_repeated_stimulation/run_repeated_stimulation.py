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
    "rate_il2": 300 * 3600 * 24 * (1e12 / (20e-6*N_A)),
    "rate_il2_restim": 300 * 3600 * 24 * (1e12 / (20e-6*N_A)),
    "deg_myc": 0.37,
    "deg_il2_restim" : 2.0,
    "K_il2_cons": 7.5, #*N_A*20e-6*10e-12,
    "K_il2" : 5,
    "K_myc": 0.1,
    "hill": 3,
    "c_il2_ex": 0,
    "up_il2": 1 * 3600 * 24 * (1e12 / (20e-6*N_A)),
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

def generate_poisson_process(mu, num_events):
    """
    generate times of restimulation
    mu is rate parameter, higher mu means more stimulation times
    """
    time_intervals = -np.log(np.random.random(num_events)) / mu
    total_events = time_intervals.cumsum()

    return total_events

def gen_poisson_start_times(mu, end_runtime=120):
    """
    create list of tuples for stepwise ode simulation
    """
    a = generate_poisson_process(mu, 200)
    a = a[a<end_runtime]
    a = np.concatenate(([0], a, [end_runtime]))
    b = [(a[i],a[i+1]) for i in range(len(a)-1)]
    assert len(b) > 0

    b = np.round(b,2)
    return b

def run_stimulations(arr, end_runtime = 120,
                     repeats = 1, stimulation_type = "poisson"):
    """
    generates start times for restimulation based either on periodic restimulation
    or random restimulation based on poisson process
    repeats: should only be used with poisson type stimulation
    arr: array of either poisson rate parametesr or spacing between stimulations
    """
    cells = []
    mols = []

    for val in arr:
        for i in range(repeats):

            # generate appropriate start times for each repeat
            if stimulation_type == "poisson":
                start_times = gen_poisson_start_times(val, end_runtime)
            else:
                assert stimulation_type == "equal_spacing"
                assert repeats == 1
                start_times = gen_start_times(val, res = 1000, end_runtime = end_runtime, end_stimulation = 30)

            print("running simulation " + str(i))
            print(start_times)

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
            df["ID"] = i

            df_mols = pd.concat([sim1.molecules_tidy, sim2.molecules_tidy, sim3.molecules_tidy]).reset_index(drop=True)
            df["dur"] = val
            df_mols["dur"] = val
            cells.append(df)
            mols.append(df_mols)

    cells = pd.concat(cells).reset_index(drop = True)
    mols = pd.concat(mols).reset_index(drop = True)

    return cells, mols


start_times = [(0,5),(5,10),(10,50),(50,51),(51,52),(52,53),(54,55), (55,56), (56,57)]
start_times = [(0,20),(20,60)]
sim1 = Simulation(name="IL2", mode=model.il2_menten_prolif, parameters=d,
                  core=model.diff_effector, start_times=start_times)

sim2 = Simulation(name="Timer", mode=model.timer_menten_prolif, parameters=d,
                  core=model.diff_effector, start_times=start_times)

sim3 = Simulation(name="Mixed", mode=model.timer_il2_menten, parameters=d,
                  core=model.diff_effector, start_times=start_times)


df_list = []
df_list2 = []
sim_list = [sim1, sim2, sim3]
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


g = sns.relplot(data = df_mols, x = "time",y = "value", col = "variable", facet_kws = {"sharey" : False},
                kind = "line", hue = "name", height = 1.8)
g.set(xlabel = "time (d)", yscale = "log", ylim = [0.1,10])
sns.despine(top=False, right=False)
plt.show()


mu = 0.01
res = 30
dur_arr = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#

stimulation_type = "poisson"
if stimulation_type == "poisson":
    dur_arr = [0.1]
    repeats = 10
else:
    repeats = 1
    dur_arr = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

for mu in [[0.1]]:
    repeats = 30
    cells, mols = run_stimulations(mu, repeats = repeats, stimulation_type= stimulation_type, end_runtime= 120)
#g = sns.relplot(data = cells, x = "time",y = "cells", kind = "line",
#                col = "name", hue = "dur", height = 2, palette = "Greys_r")
#g.set(xlim= (0,None), yscale = "log", ylim = [1,None])
#plt.show()

#cells.to_csv("repeated_stimulation_equal_spacing.csv", index = False)
    cells.to_csv("repeated_stimulation_" + str(mu[0]) + "_" + "res_" + str(repeats) + ".csv", index = False)
#g = sns.relplot(data = cells, x = "time",y = "cells", kind = "line", hue = "ID",
#                col = "name", height = 2)
#g.set(xlim= (0,None), yscale = "log", ylim = [1,None])
#plt.show()

#plt.show()

#
# g = sns.relplot(data = mols, x = "time",y = "value", row = "variable", facet_kws = {"sharey" : False},
#                 kind = "line", hue = "dur", col = "name", height = 2)
# plt.show()
#
# #
#out = cells.groupby(["dur", "name"])["cells"].agg("max").reset_index()
#out["freq"] = 1 / out["dur"]
#g = sns.relplot(data = out, x = "freq", y = "cells", hue = "name", height = 2)
#g.set(yscale = "log", ylim = [1])
#plt.show()