# -*- coding: utf-8 -*-
"""
"""
import readout_module_repeated_stimulation as readouts
import models_repeated_stimulation as model
import numpy as np
from scipy.integrate import odeint
import pandas as pd

class Simulation:
    """
    model simulation class
    initialize with a name (str), mode (model specific)
    parameters (dict), time to sim (arr) and core(model specific)
    """
    def __init__(self, name, mode, parameters, start_times = [(0,30)]):
        self.name = name
        self.mode = mode
        self.parameters = dict(parameters)
        self.default_params = dict(parameters)
        self.time = None
        self.state = None
        self.state_raw = None
        self.molecules = None
        self.start_times = start_times
        
    def init_model(self):
        """
        set initial conditions for ODE solver
        """
        n_molecules = 2 # myc and il2_ex
        y0 = np.zeros(self.parameters["alpha"]+1*self.parameters["alpha_p"]+n_molecules)
        y0[-2] = self.parameters["c_il2_ex"]

        # multiply y0 as often as there are stimulations
        y0_tile = np.tile(y0, len(self.start_times))
        # add a global IL2 concentration at the end
        il2_global_init = 0

        n_stim = len(self.start_times)
        if n_stim > 1: # set myc but only for first population
            y0_tile = np.split(y0_tile, n_stim)

            # set initial cells for each population to 1
            for j in range(n_stim):
                y0_tile[j][-1] = 1.  # regular timer
                y0_tile[j][-2] = 1.  # il2 restim timer
                y0_tile[j][0] = self.parameters["initial_cells"] # should be 1 in this case
            y0_tile = np.concatenate(y0_tile)
        else:
            y0_tile[-1] = 1
            y0_tile[-2] = 1
        y0_tile = np.concatenate((y0_tile, [il2_global_init]))
        y0_tile[0] = self.parameters["initial_cells"]
        return y0_tile

    def get_cells(self):
        """
        summarize effector cells and naive cells
        """
        teff = self.state_raw[:, self.parameters["alpha"]:] # myc and il2_ex are at end of array
        teff = np.sum(teff, axis = 1)
        # note that this is not accurate since now I have myc in the state array but I dont use naive cells anyways
        tnaive = np.sum(self.state_raw, axis = 1) - teff
        
        cells = np.stack((tnaive, teff), axis = -1)
        return cells    

    def run_ode(self):
        """
        should generate same
        run one large ODE with different starting times
        needs parameters that indicate different starting times...
        return: should return same as run_ode function
        """
        start_times = self.start_times

        y0 = self.init_model()

        mode = self.mode
        params = dict(self.parameters)

        out = []
        t_arr = []
        n_stim = len(start_times)
        for i in range(n_stim):

            tstart, tend = start_times[i]
            if i > 0:
                il2_global = y0[-1]
                y0 = y0[:-1]
                y0 = np.split(y0, n_stim)

                idx = np.random.randint(0, n_stim, 1)
                for j in idx:
                    y0[j][-1] = 1 # reset myc
                    y0[j][-2] = 1 # reset timer il2 secretion
                y0 = np.concatenate(y0)
                # set myc for new restim
                y0 = np.concatenate((y0, [il2_global]))

            time = np.arange(tstart, tend+0.01, 0.01)
            state = odeint(model.repeated_stimulation, y0, time, args=(mode, params, n_stim))
            out.append(state[:-1,:])
            y0 = state[-1,:]

            # for new initial conditions, reset timer and add initial cell population
            t_arr.append(time[:-1])

        state = np.vstack(out)
        self.time = np.concatenate(t_arr)
        # factor out global IL2 before splitting array and summing across all stimulations
        il2 = state[:,-1]  # convert to pMol
        state = state[:,:-1]
        state = np.split(state, n_stim, axis = 1)
        state = np.sum(state, axis = 0)

        cells = state[:,:-2]
        molecules = state[:,-2:]

        # reset the IL2 from individual simulations (IL2 external, not used atm)
        # with global IL2
        molecules[:,0] = il2 # * (1e12 / (20e-6*N_A)) # now also incorporates IL2 from individual populations

        self.state_raw = cells
        self.molecules = molecules
        return state

    def run_timecourse(self):

        self.run_ode()
        cells = self.get_cells()
        mols = self.molecules
        df_mols = pd.DataFrame(mols, columns = ["IL2", "Myc"])
        df_mols["time"] = self.time
        df_mols["name"] = self.name
        df_mols = df_mols.melt(id_vars= ["time", "name"])
        self.molecules_tidy = df_mols

        df = pd.DataFrame(cells, columns = ["tnaive", "cells"])
        df = df.drop(columns = ["tnaive"])
    
        df["time"] = self.time
        df["name"] = self.name

        # change this to modelname if you want to compare menten and thres
        df["model_name"] = self.mode.__name__

        self.state = df
        return df


    def get_readouts(self):
        """
        get readouts from state array
        """
        state = self.state
        peak = readouts.get_peak_height(state.time, state.cells)
        area = readouts.get_area(state.time, state.cells)
        tau = readouts.get_peaktime(state.time, state.cells)
        decay = readouts.get_duration(state.time, state.cells)
        
        reads = [peak, area, tau, decay]
        read_names = ["Peak", "Area", "Peaktime", "Decay"]
        data = {"readout" : read_names, "read_val" : reads}
        reads_df = pd.DataFrame(data = data)
        reads_df["name"] = self.name
        
        if "menten" in self.mode.__name__ :
            modelname = "menten"
        else:
            modelname =  "thres"
            
        reads_df["model_name"] = modelname
        
        return reads_df

    
    def reset_params(self):
        """
        reset parameters to default state
        """
        self.parameters = dict(self.default_params)

