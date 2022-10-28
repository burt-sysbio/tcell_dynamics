import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("paper_theme_python.mplstyle")

#df1 = pd.read_csv("repeated_stimulation_equal_spacing.csv")
df1 = pd.read_csv("repeated_stimulation_0.1_res_30.csv")

#rm_dur = [6]
#df2 = df1.loc[~df1.dur.isin(rm_dur),:]

n_stim = len(df1)
palette = ["0.3" for _ in range(df1.ID.drop_duplicates().shape[0])]
g = sns.relplot(data = df1, x = "time",y = "cells", kind = "line",
                col = "name", hue = "ID", height = 1.9, aspect = 0.9, palette = palette, alpha = 0.8,
                lw = 0.6, legend = False)
g.set(xlim= (0,None), yscale = "log", ylim = [1,1e20], xlabel = "time (d)")
sns.despine(top = False, right = False)
plt.show()

df2 = df1.loc[df1.name == "IL2"].reset_index(drop = True)
df3 = df1.loc[df1.name == "Timer"].reset_index(drop = True)
df4 = df1.loc[df1.name == "Mixed"].reset_index(drop = True)



g.savefig("repeated_stimulation.pdf")

fig, ax = plt.subplots(figsize = (2.1,1.9))

palette2 = ["tab:blue" for _ in range(df1.ID.drop_duplicates().shape[0])]
palette3 = ["tab:purple" for _ in range(df1.ID.drop_duplicates().shape[0])]
palette4 = ["0.3" for _ in range(df1.ID.drop_duplicates().shape[0])]

for df, color in zip([df2,df3,df4],[palette2, palette3, palette4]):
    sns.lineplot(data = df, x = "time", y = "cells", hue = "ID", palette = color, alpha = 1,
                    lw = 0.6, ax = ax, legend = False)

ax.set_yscale("log")
ax.set_xlabel("time (d)")
ax.set_ylabel("cells")
ax.set_ylim([1,1e20])
ax.set_xlim([0,100])
plt.tight_layout()
plt.show()

fig.savefig("repeated_stimulation_combined.pdf")


mydf = df1.loc[df1.time>60]
mydf = mydf.loc[mydf.time<60+1e-2]

g = sns.catplot(data = mydf, x = "name", y = "cells", kind = "box", height = 1.7, fliersize = 1)
g.set(yscale = "log", xlabel = "", ylabel = "cells day 60")
sns.despine(top = False, right = False)

plt.show()

g.savefig("repeated_stimulation_boxplot.pdf")



#g = sns.relplot(data = df2, x = "time",y = "cells", kind = "line",
 #               col = "name", hue = "ID", height = 2)
#g.set(xlim= (0,None), yscale = "log", ylim = [1,None])
#plt.show()