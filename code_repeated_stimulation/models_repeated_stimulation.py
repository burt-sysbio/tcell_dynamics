# -*- coding: utf-8 -*-
"""
keep all ode models here
"""
import numpy as np
from scipy.constants import N_A
# =============================================================================
# linear models
# =============================================================================
def th_cell_diff(th_state, time, model, d, il2_global):
    """

    """
    myc = th_state[-1]
    beta_p = model(myc, il2_global, time, d)
    rate_death = d["d_eff"]
    dt_state = th_cell_core(th_state, rate_death, beta_p, d, time)

    return dt_state

def th_cell_core(th_state, rate_death, beta_p, d, time):
    """
    model2
    takes state vector to differentiate effector cells as linear chain
    needs alpha and beta(r) of response time distribution, probability
    and number of precursor cells
    """
    # divide array into cell states
    myc = th_state[-1]
    il2_ex = th_state[-2]

    n_molecules = 2
    th_state = th_state[:-n_molecules]
    tnaive, tint, teff = get_cell_states(th_state, d)

    # apply feedback on rate beta
    beta = d["beta"]
    # check homeostasis criteria
    # differentiation
    dt_state = diff_effector(th_state, teff, d, beta, rate_death, beta_p)
    d_myc = -d["deg_myc"]*myc
    dt_il2_ex = -d["deg_il2_restim"]*il2_ex
    dt_state = np.concatenate((dt_state, [dt_il2_ex], [d_myc]))

    return dt_state

def generate_poisson_process(mu, num_events):
    """
    function to draw exponentially distributed inter-event times
    """
    time_intervals = -np.log(np.random.random(num_events)) / mu
    total_events = time_intervals.cumsum()

    return total_events

def repeated_stimulation(state, time, model, d, n_stim):

    # global IL2 is an extra variable!
    il2_global = state[-1]
    state = state[:-1]

    # split states according to number of stimulations
    dt_state = np.zeros_like(state)
    state_split = np.split(state, n_stim)
    dt_state_split = np.split(dt_state, n_stim)

    # dictionaries are only different in starting time other params should be the same
    # compute total effectors and consumers to then compute total IL2
    effectors = [get_cell_states(mystate[:-2], d) for mystate in state_split]
    tnaive = np.sum([x[0] for x in effectors])
    tint = np.sum([x[1] for x in effectors])
    teff = np.sum([x[2] for x in effectors])

    teff_arr = [x[2] for x in effectors]
    myc_arr = [x[-2] for x in state_split]
    # compute contributions of individual restimulated populations to IL2 secretion
    teff_restim = np.sum([x*y*d["rate_il2_restim"] for x, y in zip(teff_arr, myc_arr)])

    # if I include tnaive in the IL2 production for repeated stimulation
    # it fails because naive cells are already initialized in the beginning
    dt_il2 = d["rate_il2"] * (tint+tnaive) - \
             d["up_il2"] * (tint+teff) * (il2_global**1/(il2_global**1+d["K_il2_cons"]**1)) - \
             d["deg_il2"] * il2_global + \
             teff_restim # now the restimulation effect is also saturated
    il2_effective = il2_global

    # compute time changes but only for array where start time is reached, otherwise say zero change
    for i in range(n_stim):
        mystate = state_split[i]
        dt_state_split[i] = th_cell_diff(mystate, time, model, d, il2_effective)

    dt_state = np.hstack(dt_state_split)
    # add il2 back
    dt_state = np.concatenate((dt_state, [dt_il2]))
    return dt_state

        
def diff_effector(th_state, teff, d, beta, rate_death, beta_p):
   
    # make empty state vector should not include myc because I add it extra
    dt_state = np.zeros_like(th_state)

    # check that alpha is even number, I need this, so I can use alpha int = 0.5alpha
    assert d["alpha"] % 2 == 0
    alpha_int = int(d["alpha"] / 2)
    # calculate number of divisions for intermediate population

    n_div1 = d["n_div"]
    n_div2 = d["n_div"]
    for j in range(len(th_state)):
        #print(j)
        if j == 0:
            dt_state[j] = d["b"]-(beta+d["d_naive"])*th_state[j] 
            
        elif j < alpha_int:
            dt_state[j] = beta*th_state[j-1]-(beta+d["d_naive"])*th_state[j]
            
        elif j == alpha_int:
            dt_state[j] = n_div1*beta*th_state[j-1]-(beta+d["d_prec"])*th_state[j]

        elif j < (d["alpha"]):
            dt_state[j] = beta*th_state[j-1]-(beta+d["d_prec"])*th_state[j]
        
        elif j == (d["alpha"]):
            dt_state[j] = n_div2*beta*th_state[j-1] + (2*beta_p*th_state[-1]) - (rate_death+beta_p)*th_state[j]       
        
        else:
            dt_state[j] = beta_p*th_state[j-1]-(beta_p+rate_death)*th_state[j]
        
    return dt_state

# =============================================================================
# helper functions
# =============================================================================
def get_cell_states(th_state, d):
    assert d["alpha"] % 2 == 0
    assert d["alpha"] > 0
    
    alpha_int = int(d["alpha"] / 2)
    
    # this is for the alpha int model
    tnaive = th_state[:alpha_int]
    tint = th_state[alpha_int:d["alpha"]]
    teff = th_state[d["alpha"]:]

    tnaive = np.sum(tnaive)
    tint = np.sum(tint)
    teff = np.sum(teff)    
    
    return tnaive, tint, teff


##############################################################################################
##############################################################################################
##############################################################################################
# proliferation models

def menten(conc, vmax, K, hill):
    # make sure to avoid numeric errors for menten
    out = (vmax*conc**hill) / (K**hill+conc**hill)
    return out


def il2_menten_prolif(myc, il2, time, d):

    vmax = d["beta_p"]
    beta_p = menten(il2, vmax, d["K_il2"], d["hill"])

    return beta_p


def timer_menten_prolif(myc, il2, time, d):
    vmax = d["beta_p"]
    K = d["K_myc"]
    hill = d["hill"]

    beta_p = menten(myc, vmax, K, hill)

    return beta_p

def timer_il2_menten(myc, il2, time, d):

    vmax = d["beta_p"]
    K = d["K_myc"]
    hill = d["hill"]
    # compute beta p as product of il2 and myc effect
    beta_p = d["beta_p"] * menten(myc, 1.0, K, hill) * menten(il2, 1.0, d["K_il2"], hill)

    return beta_p

