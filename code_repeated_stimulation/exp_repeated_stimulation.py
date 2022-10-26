# -*- coding: utf-8 -*-
"""
"""
import readout_module_repeated_stimulation as readouts
import models_repeated_stimulation as model

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import copy
import itertools
import matplotlib.ticker as ticker
from scipy.optimize import minimize_scalar
from matplotlib.colors import LogNorm
from scipy.stats import lognorm as log_pdf
import warnings
from scipy.constants import N_A

def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale


def change_param(simlist, pname, arr):
    assert len(arr) == len(simlist)
    for sim, val in zip(simlist,  arr):
        #print(val)
        sim.name = val
        sim.parameters[pname] = val
    
    return simlist

    
def make_sim_list(Simulation, n = 20):
    sim_list = [copy.deepcopy(Simulation) for i in range(n)]
    return sim_list


class Simulation:
    """
    model simulation class
    initialize with a name (str), mode (model specific)
    parameters (dict), time to sim (arr) and core(model specific)
    """
    def __init__(self, name, mode, parameters, core, start_times = [(0,30)]):
        self.name = name
        self.mode = mode
        self.parameters = dict(parameters)
        self.time = None
        self.core = core
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

    def get_il2_ex(self):
        assert self.molecules is not None
        il2 = self.molecules[:, 0]
        return il2


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
        core = self.core

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

            time = np.arange(tstart, tend, 0.01)
            state = odeint(model.repeated_stimulation, y0, time, args=(mode, params, core, n_stim))
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
        molecules[:,0] = il2 * (1e12 / (20e-6*N_A)) # now also incorporates IL2 from individual populations

        self.state_raw = cells
        self.molecules = molecules
        return state

    def run_timecourse(self, normalize = False):
        if normalize == True: warnings.warn("warning: normalize is True")

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
    
        # normalize cell area
        if normalize == True:
            area = np.trapz(df.cells.array, self.time)
            df.cells = df.cells.div(area) 
  
        # change this to modelname if you want to compare menten and thres
        df["model_name"] = self.mode.__name__

        self.state = df
        return df


    def get_il2_max(self):
        """
        get il2 concentration over time excluding external IL2
        return IL2 total concentration and il2 external concentration
        """
        if self.state is None:
            self.run_timecourse()
            print("no timecourse run yet, running timecourse...")

        d = dict(self.parameters)
        time = self.time
        state = self.state_raw
        alpha_int = int(d["alpha"] / 2)
        
        # get intermediate cell population and sum it up over all int states
        tint = state[:, alpha_int:d["alpha"]]
        il2_producers = np.sum(tint, axis = 1)
        
        il2_consumers = state[:, d["alpha"]:]
        il2_consumers = np.sum(il2_consumers, axis = 1)
        #get external il2_conc as array

        il2 = d["rate_il2"]*il2_producers / (d["K_il2"]+il2_consumers)
        # get maximum il2 production per second (IL2 producer max * IL2 rate)
        #il2_max = np.max(tint)*params["rate_il2"] + params["c_il2_ex"]
        
        return il2
    
    
    def plot_timecourse(self):
        g = sns.relplot(data = self.state, x = "time", y = "cells", kind = "line")
        return g
    

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

            
    def vary_param(self, pname, arr, normtype = "first"):
        old_parameters = dict(self.parameters)
        readout_list = []
        edge_names = ["alpha", "alpha_p"]
        #print(pname)
        # edgecase for distributions
        dummy = None
        if pname in edge_names:
            dummy = "beta" if pname == "alpha" else "beta_p"
            arr = np.arange(2, 20, 2)

        for val in arr:
            # edgecase for distributions
            if pname in edge_names:
                self.parameters[dummy] = val
                
            self.parameters[pname] = val 
            self.run_timecourse()    
            read = self.get_readouts()
            #print(val)
            #print(read)
            read["p_val"] = val
            readout_list.append(read)

        self.parameters = old_parameters      
        df = self.vary_param_norm(readout_list, arr, edge_names, normtype, pname)        
        return df


    def vary_param_norm(self, readout_list, arr, edge_names, normtype, pname):
        """
        take readout list and normalize to middle or beginning of array
        Parameters
        ----------
        readout_list : list
            readouts for diff param values.
        arr : array
            parameter values.
        edgenames : list of strings
            parameter names.
        normtype : string, should be either "first" or "middle"
            normalize to middle or beginning of arr.

        Returns
        -------
        df : data frame
            normalized readouts
        """
        
        df = pd.concat(readout_list)
        df = df.reset_index(drop = True)
        
        # merge df with normalization df    
        norm = arr[int(len(arr)/2)]
        if normtype == "first":
            norm = arr[0]
        else:
            # get the value in arr, which is closest to the median
            norm = arr[np.argmin(np.abs(arr-np.median(arr)))]

        df2 = df[df.p_val == norm]

        df2 = df2.rename(columns = {"read_val" : "ynorm"})
        df2 = df2.drop(columns = ["p_val"])
        df = df.merge(df2, on=['readout', 'name', "model_name"], how='left')
        
        # compute log2FC
        logseries = df["read_val"]/df["ynorm"]
        logseries = logseries.astype(float)

        df["log2FC"] = np.log2(logseries)
        df = df.drop(columns = ["ynorm"])
        
        # add xnorm column to normalise x axis for param scans
        df["xnorm"] = df["p_val"] / norm
        df["pname"] = pname
        
        if pname in edge_names:
            df["p_val"] = df["p_val"] / (df["p_val"]*df["p_val"])
            
        return df
    
    
    def norm(self, val, pname, norm):
        """
        optimization function
        calculate difference between simulated response size and wanted response size
        val : parameter value
        pname: str, parameter name
        norm : wanted response size
        returns float, difference in abs. values between wanted resp. size and calc. response size
        """
        self.parameters[pname] = float(val)
        state = self.state
        area = readouts.get_area(state.time, state.cells)

        return np.abs(area-norm)

    
    def norm_readout(self, pname, norm, bounds = None):
        """
        adjust parameter to given normalization condition (area = norm)
        pname: str of parameter name to normalize
        norm : value of area to normalize against
        bounds : does not work well - if bounds provided, only scan for paramter in given range
        returns: adjusted parameter value
        """
        if bounds != None:
            method = "Bounded"
        else:
            method = "Brent"
        
        # minimize norm function for provided values
        out = minimize_scalar(self.norm, method = method, bounds=bounds, args=(pname, norm, ))  
        #if pname == "beta_p":
        #    print(out)
        #    print(self.mode)
        
        dummy = np.nan
        print(out.success, out.fun, out.x)
        # check results of optimization object
        if out.success == True and out.fun < 1e-2:
            dummy = out.x
            
        return dummy


    def get_heatmap(self, arr1, arr2, name1, name2, norm_list = None):
        
        """
        make a heatmap provide two arrays and two parameter names as well
        as readout type by providing readout function
        can also provide normalization value for log2 presentation
        """
        area_grid = []
        peaktime_grid = []
        peak_grid = []
        grids = [area_grid, peaktime_grid, peak_grid]
        readout_funs = [readouts.get_area, readouts.get_peaktime, readouts.get_peak_height]
        old_params = dict(self.parameters)

        assert (len(readout_funs) == len(grids))
        # if no normalization was provided do not normalize so use None for each readout
        if norm_list is None:
            norm_list = [None for _ in readout_funs]

        for val1, val2 in itertools.product(arr1, arr2):
            # loop over each parameter combination and get readouts
            self.parameters[name1] = val1
            self.parameters[name2] = val2
            df = self.run_timecourse()   
            
            # get each readout and normalize, append to grid
            for readout_fun, grid, norm_val in zip(readout_funs, grids, norm_list):
                readout = readout_fun(df.time, df.cells)
                if norm_val != None:
                    readout = np.log2(readout/norm_val)
                grid.append(readout)

            self.parameters = dict(old_params)
            #print(len(z))

        # process output, combine readouts
        df = pd.DataFrame(grids)
        df = df.T
        df.columns = ["Area", "Peak Time", "Peak Height"]

        mylist = [(x, y) for x, y in itertools.product(arr1, arr2)]
        df[["param_val1", "param_val2"]] = mylist
        df=df.melt(id_vars=["param_val1", "param_val2"], var_name = "readout")
        df["name"] = self.name
        df["pname1"] = name1
        df["pname2"] = name2
        return df
 
    
    def plot_heatmap(self, arr1, arr2, name1, name2, readout_fun, norm = None, 
                     vmin = None, vmax = None, title = None, 
                     label1 = None, label2 = None, cmap = "bwr", log = True,
                     cbar_label = "change response size"):
    
        arr1, arr2, val = self.get_heatmap(arr1, arr2, name1, name2, readout_fun, norm)

        fig, ax = plt.subplots(figsize = (6,4))
        color = cmap
        cmap = ax.pcolormesh(arr1, arr2, val, cmap = color, vmin = vmin, vmax = vmax,
                             rasterized = True)

        
        loc_major = ticker.LogLocator(base = 10.0, numticks = 100)
        loc_minor = ticker.LogLocator(base = 10.0, 
                                      subs = np.arange(0.1,1,0.1),
                                      numticks = 12) 
        
        if log == True:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(loc_major)
            ax.xaxis.set_minor_locator(loc_minor)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_title(title)
        cbar = plt.colorbar(cmap)
        cbar.set_label(cbar_label)
            
        plt.tight_layout()
    
        return fig
    
    def gen_lognorm_params(self, pname, std, n = 20): 
        mean = self.parameters[pname]
        sigma, scale = lognorm_params(mean, std)
        sample = log_pdf.rvs(sigma, 0, scale, size = n)
        
        return sample
    
    def draw_new_params(self, param_names, heterogeneity):
        for param in param_names:
            mean = self.parameters[param] 
            std = mean*(heterogeneity/100.)
            sigma, scale = lognorm_params(mean, std)
            sample = log_pdf.rvs(sigma, 0, scale, size = 1)
            self.parameters[param] = sample
# =============================================================================
# end class
# =============================================================================
    
def gen_arr(sim, pname, scales = (1,1), n = 30):
    edge_names = ["alpha", "alpha_1", "alpha_p"]
    if pname in edge_names:
        arr = np.arange(2, 20, 2)
    else:
        params = sim.parameters
        val = params[pname]
        val_min = 10**(-scales[0])*val
        val_max = 10**scales[1]*val
        arr = np.geomspace(val_min, val_max, n)
    return arr
        

class SimList:
       
    def __init__(self, sim_list):
        self.sim_list = sim_list
 
    
    def reduce_list(self, cond):
        sim_list_red = [sim for sim in self.sim_list if np.abs(sim.get_area()-cond) < 1.]
        return SimList(sim_list_red)
    
    
    def get_readout(self, name):
        readout_list = []
        for sim in self.sim_list:
            df = sim.get_readouts()
            # check that readout name is actually available since I change readouts sometimes
            assert name in df.readout.values
            out = float(df.read_val[df.readout == name])
            readout_list.append(out)
        return readout_list
    
    
    def get_tau(self):       
        return self.get_readout("tau")
 
    
    def get_peak(self):
        return self.get_readout("Peak")


    def get_area(self):        
        return self.get_readout("Area")
   
         
    def pscan(self, pnames, arr = None, scales = (1,1), n = 30, normtype = "first"):
        pscan_list = []
        for sim in self.sim_list:
            for pname in pnames:
                if arr is None:
                    assert len(arr) == n
                    arr = gen_arr(sim = sim, pname = pname, scales = scales, n = n)
                else:
                    assert len(arr) == n
                readout_list = sim.vary_param(pname, arr, normtype)
                
                pscan_list.append(readout_list)
        
        df = pd.concat(pscan_list)
        return df
    
    
    def run_timecourses(self):

        df_list = [sim.run_timecourse() for sim in self.sim_list]
        df = pd.concat(df_list)
        return df
    
    
    def normalize(self, pname, norm, bounds):
        out_list = []
        for sim in self.sim_list:
            out = sim.norm_readout(pname, norm, bounds = bounds)
            out_list.append(out)
        return out_list


    def get_param_arr(self, pname):
        out = [sim.parameters[pname] for sim in self.sim_list]
        return out


    def get_il2_max(self):
        
        il2_max = [sim.get_il2_max() for sim in self.sim_list]
        return il2_max
    
    
    def plot_timecourses(self, arr, arr_name, log = True, log_scale = False, xlim = (None, None),
                         ylim = (None, None), cmap = "Greys", cbar_scale = 1., 
                         il2_max = False, ticks = None, data = None):
        """
        plot multple timecourses with colorbar
        can provide a data argument from run timecourses
        """
        # parameter for scaling of color palette in sns plot
        hue_norm = None
        if log == True:
            vmin = np.min(arr)
            vmax = np.max(arr)
            norm = matplotlib.colors.LogNorm(
                    vmin = vmin,
                    vmax = vmax)
            # if log then also in sns plot
            hue_norm = LogNorm(vmin = vmin, vmax = vmax)

        else:
            norm = matplotlib.colors.Normalize(
                    vmin=np.min(arr),
                    vmax=np.max(arr))  
        
        # make mappable for colorbar
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # run the time courses to generate data        
        if data is None:
            print("hi there")
            data = self.run_timecourses()
        
        # hue takes the model name, so this should be a scalar variable
        # can be generated by change_param function
        g = sns.relplot(x = "time", y = "cells", kind = "line", data = data, hue = "name", 
                        hue_norm = hue_norm, col = "model_name", palette = cmap, 
                        height = 5,legend = False, aspect = 1.2, 
                        facet_kws = {"despine" : False})

        
        g.set(xlim = xlim, ylim = ylim)
        ax = g.axes[0][0]
        ax.set_ylabel("cell dens. norm.")
        g.set_titles("{col_name}")
        
        # if ticks are true take the upper lower and middle part as ticks
        # for colorbar
        if ticks == True:
            if log == True:
                ticks = np.geomspace(np.min(arr), np.max(arr), 3)
            else:
                ticks = np.linspace(np.min(arr), np.max(arr), 3)

            cbar = g.fig.colorbar(sm, ax = g.axes, ticks = ticks)
            cbar.ax.set_yticklabels(np.round(cbar_scale*ticks,2))
        else:
            cbar = g.fig.colorbar(sm, ax = g.axes, ticks = ticks)
        # add colorbar
        
        cbar.set_label(arr_name)
        #print(100*arr)
        
        
        if il2_max == True:
            # get max il2 concentrations
            il2_arr = self.get_il2_max()
            lower = np.min(il2_arr)
            upper = np.max(il2_arr)
            
            # split il2 array in lower upper and middle to assign ticks
            labels = np.geomspace(lower, upper, 3) if log == True else np.linspace(lower, upper, 3)
            print(il2_arr)
            # combine with external il2 concentration (stored in ticks)
            labels = ticks/labels
            #cbar.ax.set_yticklabels(cbar_scale*labels)
            cbar.ax.set_yticklabels(np.round(cbar_scale*labels,2))
        
        cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        
        if log_scale == True:
            g.set(yscale = "log", ylim = (0.1, None))    
        

        return g, data
    
    
    def plot_pscan(self, pnames):
        data = self.pscan(pnames)
        g = sns.relplot(data = data, x = "xnorm", hue = "model_name", y = "log2FC", col = "readout",
                    row = "pname", kind = "line")
        g.set(xscale = "log")
        
        return g


