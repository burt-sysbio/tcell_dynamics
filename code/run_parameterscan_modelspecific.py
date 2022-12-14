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

out_list = []

pnames = ["rate_il2_prec", "up_il2", "rate_il2_naive"]
max_step = 0.01
for pname in pnames:
    arr = sim_il2.gen_arr(pname = pname, use_percent = True, scales = (0.9,1.1))
    out = sim_il2.vary_param(pname = pname, arr = arr, normtype="middle", t_eval = np.arange(0,80,0.1), max_step = max_step)
    out_list.append(out)

pname = "deg_myc"
arr = sim_timer.gen_arr(pname = pname, use_percent = True, scales = (0.9,1.1))
out1 = sim_timer.vary_param(pname = pname, arr = arr, normtype= "middle", t_eval = np.arange(0,80,0.1), max_step = max_step)

pname = "n_crit"
arr = sim_carry.gen_arr(pname = pname, use_percent = True, scales = (0.9,1.1))
out2 = sim_carry.vary_param(pname = pname, arr = arr, normtype= "middle", t_eval = np.arange(0,80,0.1), max_step = max_step)


out_il2 = pd.concat(out_list).reset_index()


out_il2.to_csv("../output/paramscans/pscan_il2_specific.csv", index = False)
out1.to_csv("../output/paramscans/pscan_timer_specific.csv", index = False)
out2.to_csv("../output/paramscans/pscan_carry_specific.csv", index = False)


