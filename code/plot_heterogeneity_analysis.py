import numpy as np
import pandas as pd
import seaborn as sns

from exp import Simulation, SimList
import models as model
from params_fig2 import d
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("paper_theme_python.mplstyle")
sns.set_palette("deep")
CV_list = [0.01,0.05,0.1,0.5,1.0]

mylist = []


time_il2 = [(0, 100)]
time_timer = [(0, 100)]

t_eval = np.arange(0,80,0.1)

def vir_model_const(t):
    return 0


sim_il2 = Simulation(name="IL2", mode=model.il2_prolif, parameters=d, start_times=time_il2,
                  vir_model=vir_model_const)

sim_timer = Simulation(name="Timer", mode=model.timer_prolif, parameters=d, start_times=time_timer,
                  vir_model=vir_model_const)

cells_il2 = sim_il2.compute_cellstates(t_eval = t_eval, max_step = np.inf)

sim_timer.parameters["deg_myc"] = 0.395
cells_timer = sim_timer.compute_cellstates(t_eval = t_eval, max_step = np.inf)

cells_il2 = cells_il2.loc[cells_il2.species == "CD4_all"]
cells_timer = cells_timer.loc[cells_timer.species == "CD4_all"]


# plot allparams or prolif
myplot = "allparams"
for CV in CV_list:
    df1 = pd.read_csv("../output/heterogeneity/heterogeneity_cv_" + str(CV) + "_IL2_" + myplot + ".csv")
    df2 = pd.read_csv("../output/heterogeneity/heterogeneity_cv_" + str(CV) + "_Timer_" + myplot + ".csv")
    mylist.append(df1)
    mylist.append(df2)


df_all = pd.concat(mylist)

df_all = df_all.loc[df_all.species == "CD4_all"]
df_all.reset_index(drop=True, inplace=True)

# plot heterogeneity timecourses

CV_red = [0.01, 0.1]
df_red = df_all.loc[df_all.CV.isin(CV_red)]
g = sns.relplot(data = df_red, x = "time", y= "value", kind = "line", ci = "sd",
                hue = "name", col = "CV", height = 1.8, aspect = 0.9, facet_kws = {"sharey" : True})
g.set(xlabel = "time (h)", ylabel = "cells")

# ax1 = g.axes[0][0]
# ax2 = g.axes[0][1]
# sns.lineplot(data = cells_il2, x = "time", y = "value", ax = ax1)
# sns.lineplot(data = cells_timer, x = "time", y = "value", ax = ax2)

sns.despine(top = False, right = False)
plt.show()

sname = "../figures/supplements/supp_heterogeneity"
g.savefig(sname + ".pdf")
g.savefig(sname + ".svg")


# plot CV vs max

df_max = df_all.groupby(["run_ID", "name", "CV"])["value"].agg("max").reset_index()

g = sns.catplot(data = df_max, x = "CV", y = "value", hue = "name", kind = "strip", dodge = True,
                height = 1.8, s = 2)
g.set(ylim = [1, 1e10], yscale = "log", ylabel = "response maximum")
plt.show()
