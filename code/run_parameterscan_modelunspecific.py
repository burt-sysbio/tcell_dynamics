#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:14:25 2020

@author: burt
"""

from exp import Simulation, SimList
import models as model
from params_fig2 import d
import numpy as np
import pandas as pd

d = dict(d)
# =============================================================================
# make time course
# =============================================================================
import os

print(os.getcwd())
# for carrying capacacity I need a long long timecourse
# because the peak comes so late

# for timer and IL2 timecourse needs to be a bit longer
# for the cells to return to 0 at the end because they have a high peak
time_arr = [(0, 200)]


def vir_model_const(t):
    return 0


sim_il2 = Simulation(name="IL2", mode=model.il2_prolif, parameters=d, start_times=time_arr,vir_model=vir_model_const)
sim_timer = Simulation(name="Timer", mode=model.timer_prolif, parameters=d, start_times=time_arr,vir_model=vir_model_const)
sim_carry = Simulation(name="Carry", mode=model.carry_prolif, parameters=d, start_times=time_arr,vir_model=vir_model_const)
# =============================================================================
# make parameter scan
# =============================================================================

out_list1 = []
out_list2 = []
out_list3 = []

pnames = ["beta_naive", "beta_prec", "d_eff"]
max_step = 0.01
for pname in pnames:
    arr = sim_il2.gen_arr(pname = pname, use_percent = True, scales = (0.9,1.1))
    out1 = sim_il2.vary_param(pname = pname, arr = arr, normtype="middle", t_eval = np.arange(0,80,0.1), max_step = max_step)
    out_list1.append(out1)

    out2 = sim_timer.vary_param(pname = pname, arr = arr, normtype="middle", t_eval = np.arange(0,80,0.1), max_step = max_step)
    out_list2.append(out2)

    out3 = sim_carry.vary_param(pname = pname, arr = arr, normtype="middle", t_eval = np.arange(0,80,0.1), max_step = max_step)
    out_list3.append(out3)

df1 = pd.concat(out_list1).reset_index()
df2 = pd.concat(out_list2).reset_index()
df3 = pd.concat(out_list3).reset_index()

df1.to_csv("../output/paramscans/pscan_il2_unspecific.csv", index = False)
df2.to_csv("../output/paramscans/pscan_timer_unspecific.csv", index = False)
df3.to_csv("../output/paramscans/pscan_carry_unspecific.csv", index = False)
