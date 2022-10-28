import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("paper_theme_python.mplstyle")

#df1 = pd.read_csv("repeated_stimulation_equal_spacing.csv")
df1 = pd.read_csv("repeated_stimulation_0.01_res_30.csv")

#rm_dur = [6]
#df2 = df1.loc[~df1.dur.isin(rm_dur),:]
g = sns.relplot(data = df1, x = "time",y = "cells", kind = "line",
                col = "name", hue = "ID", height = 1.9, aspect = 0.9)
g.set(xlim= (0,None), yscale = "log", ylim = [1,1e20])
plt.show()

g.savefig("repeated_stimulation.pdf")
#g = sns.relplot(data = df2, x = "time",y = "cells", kind = "line",
 #               col = "name", hue = "ID", height = 2)
#g.set(xlim= (0,None), yscale = "log", ylim = [1,None])
#plt.show()