#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:14:25 2020

@author: burt
"""

from exp_repeated_stimulation import Simulation
import models_repeated_stimulation as model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("../paper_theme_python.mplstyle")
sns.set_palette("deep")
from scipy.constants import N_A

SEC_TO_DAYS = 60*60*24
MOLECULES_TO_MOLAR = (1e12 / (20e-6*N_A)) # converting to picomolar assuming lymph node volume of 20µl
IL2_SECRETION = 150

d = {
    "b": 0,
    "initial_cells": 1.0,
    "alpha": 4, # Polonsky 2018
    "beta": 2., # Polonsky 2018
    "alpha_p": 2,# Polonsky 2018
    "beta_p": 4.5,# Polonsky 2018
    "d_prec": 0,
    "d_naive": 0,
    "d_eff" : 0.24, # Zaretsky 2012
    "n_div": 2,
    "rate_il2": IL2_SECRETION * SEC_TO_DAYS * MOLECULES_TO_MOLAR, # see Huang et al 2013
    "rate_il2_restim": IL2_SECRETION * SEC_TO_DAYS * MOLECULES_TO_MOLAR, # Huang et al 2013
    "deg_myc": 0.37,
    "deg_il2_restim" : 2.0, # helmstetter 2015
    "K_il2_cons": 7.5, #*N_A*20e-6*10e-12,
    "K_il2" : 5,
    "K_myc": 0.1,
    "hill": 3,
    "c_il2_ex": 0,
    "up_il2": 1 * SEC_TO_DAYS * MOLECULES_TO_MOLAR,
    "deg_il2" : 0
}

# Huang,..., Davis (Immunity 2013): IL2 secretion rates are between 1e3 and 5e4 molecules per minute --> 16-830 molecules per cell per second
# Altan-Bonnet review 2019: uptake is restricted to 1 molecule per cell per second
# IL2 secretion after restimulation + withdrawal of stimulus lasts around 8 hours --> IFNg Helmstetter,..., Löhning Immunity 2015

# if in doubt, use long simulation times so that cell numbers return to zero

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

def run_stimulations(arr, end_runtime = 120, repeats = 1):
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

            start_times = gen_poisson_start_times(val, end_runtime)

            print("running simulation " + str(i))
            print(start_times)

            sim1 = Simulation(name="IL2", mode=model.il2_menten_prolif, parameters=d,
                              start_times = start_times)

            sim2 = Simulation(name="Timer", mode=model.timer_menten_prolif, parameters=d,
                              start_times = start_times)

            sim3 = Simulation(name="Mixed", mode=model.timer_il2_menten, parameters=d,
                              start_times = start_times)

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
                  start_times=start_times)

sim2 = Simulation(name="Timer", mode=model.timer_menten_prolif, parameters=d,
                  start_times=start_times)

sim3 = Simulation(name="Mixed", mode=model.timer_il2_menten, parameters=d,
                  start_times=start_times)


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

for mu in [[0.1]]:
    repeats = 30
    cells, mols = run_stimulations(mu, repeats = repeats, end_runtime= 120)
    cells.to_csv("repeated_stimulation_" + str(mu[0]) + "_" + "res_" + str(repeats) + ".csv", index = False)
