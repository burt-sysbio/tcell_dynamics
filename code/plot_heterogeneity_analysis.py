import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10
import seaborn as sns

sns.set_palette("deep")
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
CV_list = ["0.01", "0.05", "0.5"]
mylist =  []

# plot allparams or prolif
myplot = "allparams"
for CV in CV_list:
    df1 = pd.read_csv("../output/heterogeneity_cv_" + CV + "_IL2_" + myplot + ".csv")
    df2 = pd.read_csv("../output/heterogeneity_cv_" + CV + "_Timer_" + myplot + ".csv")
    mylist.append(df1)
    mylist.append(df2)


df_all = pd.concat(mylist)

df_all = df_all.loc[df_all.species == "CD4_all"]
df_all.reset_index(drop=True, inplace=True)

g = sns.relplot(data = df_all, x = "time", y= "value", kind = "line", ci = "sd",
                hue = "name", col = "CV", height = 2)
g.set(yscale = "log", ylim = [1, None], xlabel = "time (h)", ylabel = "cells")
sns.despine(top = False, right = False)
plt.show()

sname = "../figures/supplements/supp_heterogeneity"
g.savefig(sname + ".pdf")
g.savefig(sname + ".svg")
