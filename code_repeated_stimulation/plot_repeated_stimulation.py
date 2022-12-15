import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import colors
plt.style.use("paper_theme_python.mplstyle")

#df1 = pd.read_csv("repeated_stimulation_equal_spacing.csv")
df1 = pd.read_csv("output_repeated_stimulation/repeated_stimulation_0.1_res_50.csv")

test = df1.loc[(df1.name == "Timer") & (df1.cells > 1e40)] # filter out that weird timer numerics peak
df1 = df1.loc[~((df1.name == "Timer") & (df1.ID == 49))].copy()


n_stim = len(df1)
palette = ["0.3" for _ in range(df1.ID.drop_duplicates().shape[0])]


df_grouped = df1.groupby(["name", "ID"]).apply(lambda x: np.trapz(x.cells, x.time)).reset_index()
df_grouped.columns = ["name", "ID", "area"]
mymax = df_grouped.groupby(["name"]).area.max().reset_index()
mymin = df_grouped.groupby(["name"]).area.min().reset_index()

ID_min = pd.merge(df_grouped, mymin, how = "inner", on = ["name", "area"])
ID_max = pd.merge(df_grouped, mymax, how = "inner", on = ["name", "area"])

df_filled = pd.concat([ID_min, ID_max]).reset_index(drop = True)
df_filled2 = pd.merge(df1, df_filled, how = "inner", on = ["ID", "name"])

models = ["IL2", "Timer", "Mixed"]


mypal = sns.color_palette("deep")
mypal = [mypal[4],mypal[-2],mypal[-3]]
fig, ax = plt.subplots(figsize = (2,1.8))
time_arr = np.arange(0,60,0.1)

edgecolor = colors.to_rgba("k", alpha = 0)
alpha = 0.7
for model, color in zip(models, mypal):

    mydf = df_filled2.loc[df_filled2.name == model]
    ID_list = mydf.ID.drop_duplicates().values
    mydf_min = mydf.loc[mydf.ID == ID_list[0]]
    mydf_max = mydf.loc[mydf.ID == ID_list[1]]

    f1 = interp1d(mydf_min.time, mydf_min.cells)
    f2 = interp1d(mydf_max.time, mydf_max.cells)

    cells_max = f2(time_arr)
    cells_min = f1(time_arr)

    ax.fill_between(time_arr, cells_min, cells_max,
                    facecolor = colors.to_rgba(color, alpha), edgecolor = edgecolor)

ax.set_yscale("log")
ax.set_ylim([1,1e20])
ax.set_xlim(time_arr[0], time_arr[-1])
ax.set_xlabel("time (d)")
ax.set_ylabel("cells")
ax.set_xticks([0,30,60])
plt.tight_layout()
plt.show()
fig.savefig("repeated_stimulation_filled.pdf")
fig.savefig("repeated_stimulation_filled.svg")

g = sns.relplot(data = df1, x = "time",y = "cells", kind = "line",
                col = "name", hue = "ID", height = 1.7, aspect = 0.8, palette = palette, alpha = 0.6,
                lw = 0.6, legend = False)
g.set(xlim= (0,60), yscale = "log", ylim = [1,1e20], xlabel = "time (d)", xticks = [0,30,60])
sns.despine(top = False, right = False)
g.set_titles("{col_name}")
plt.show()

df2 = df1.loc[df1.name == "IL2"].reset_index(drop = True)
df3 = df1.loc[df1.name == "Timer"].reset_index(drop = True)
df4 = df1.loc[df1.name == "Mixed"].reset_index(drop = True)

g.savefig("repeated_stimulation.pdf")
g.savefig("repeated_stimulation.svg")


fig, ax = plt.subplots(figsize = (2.1,1.9))

palette2 = ["tab:blue" for _ in range(df1.ID.drop_duplicates().shape[0])]
palette3 = ["tab:purple" for _ in range(df1.ID.drop_duplicates().shape[0])]
palette4 = ["0.3" for _ in range(df1.ID.drop_duplicates().shape[0])]

for df, color in zip([df2,df3,df4],[palette2, palette3, palette4]):
    sns.lineplot(data = df, x = "time", y = "cells", hue = "ID", palette = color, alpha = 1,
                    lw = 0.2, ax = ax, legend = False)

ax.set_yscale("log")
ax.set_xlabel("time (d)")
ax.set_ylabel("cells")
ax.set_ylim([1,1e20])
ax.set_xlim([0,100])
plt.tight_layout()
plt.show()

fig.savefig("figures_repeated_stimulation/repeated_stimulation_combined.pdf")
fig.savefig("figures_repeated_stimulation/repeated_stimulation_combined.svg")


mydf = df1.loc[df1.time>60]
mydf2 = mydf.loc[mydf.time<60+1e-2]

PROPS = {
    'boxprops':{'edgecolor':'0.1', 'alpha' : 1},
    'medianprops':{'color':'0.1'},
    'whiskerprops':{'color':'0.1'},
    'capprops':{'color':'0.1'}
}
# 'facecolor':'0.6', can be added to PROPS
g = sns.catplot(data = mydf2, x = "name", y = "cells", kind = "box", height = 1.7, fliersize = 1,
                palette = mypal, aspect = 1.0, **PROPS)
g.set(yscale = "log", xlabel = "", ylabel = "cells day 60")
sns.despine(top = False, right = False)
g.set_xticklabels(rotation = 90)
plt.tight_layout()
plt.show()

g.savefig("figures_repeated_stimulation/repeated_stimulation_boxplot.pdf")
g.savefig("figures_repeated_stimulation/repeated_stimulation_boxplot.svg")
