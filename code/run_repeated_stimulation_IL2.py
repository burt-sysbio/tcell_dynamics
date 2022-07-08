#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:14:25 2020

@author: burt
"""
import sys
sys.path.append("../../")
from final_figures.code_final_figures.exp_final_figures import Simulation
import final_figures.code_final_figures.models as model
from final_figures.code_final_figures.params_fig2 import d
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

d = dict(d)
d["initial_cells"] = 1
d["deg_myc"] = 0.2 # make it a slow timer so that timer only kicks in once IL2 mechanism fails
#d["deg_myc"] = 1.0 # this is the timer for restimulation

def vir_model_const(t):
    return 0
# for carrying capacacity I need a long long timecourse
# because the peak comes so late

# for timer and IL2 timecourse needs to be a bit longer
# for the cells to return to 0 at the end because they have a high peak

mu_event = 3
tstart = 0
perturb_end = 30
tend = 80

def generate_start_times(mu, perturb_end = 20, tstart = 0, tend = 60):
    n_events = 100000

    events = model.generate_poisson_process(mu, n_events)
    events = np.round(events, 2)
    events = np.unique(events)
    events = events[events < perturb_end]

    if len(events) == 0:
        events = [(tstart, tend)]
    else:
        events = np.concatenate(([tstart], events))
        events = [(events[i], events[i+1]) for i in range(len(events)-1)]

        events.append((events[-1][1], tend))
    return events


res = 30
mu_arr = [0.1, 0.5, 1.0]
mu_arr = np.geomspace(0.01,1,res)
nsim = 100
mylist_il2 = []
mylist_timer_il2 = []
mylists = [mylist_il2, mylist_timer_il2]
mymodels = [model.restim, model.restim_timer_il2]
mynames = ["IL2", "IL2_Timer"]

for mu in mu_arr:
    for i in range(nsim):
        events = generate_start_times(mu, perturb_end = perturb_end, tstart = tstart, tend = tend)

        for mylist, mymodel, myname in zip(mylists, mymodels, mynames):
            sim = Simulation(name=myname, mode=mymodel, parameters=d,
                             start_times=events, vir_model=vir_model_const)
            out = sim.compute_cellstates()
            # times are differently sampled bc of events, needs binning or crude adding

            out = out.loc[out["species"] == "CD4_all"].copy()
            out["mu"] = mu
            out["ID"] = i

            mylist.append(out)

for mylist, myname in zip(mylists, mynames):
    cells = pd.concat(mylist)
    cells.to_csv("../output/repeated_stimulation_" + myname + ".csv", index = False)

