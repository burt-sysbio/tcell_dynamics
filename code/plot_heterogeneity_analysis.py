import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context = "poster", style = "ticks")
print("hello world")

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
                hue = "name", col = "CV")
g.set(yscale = "log", ylim = [1, None])
plt.show()


