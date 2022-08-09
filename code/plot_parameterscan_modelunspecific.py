from exp import Simulation, SimList
import models as model
from params_fig2 import d
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


out_il2 = pd.read_csv("../output/paramscans/pscan_il2_unspecific.csv")
out1 = pd.read_csv("../output/paramscans/pscan_timer_unspecific.csv")
out2 = pd.read_csv("../output/paramscans/pscan_carry_unspecific.csv")

df = pd.concat([out_il2, out1, out2]).reset_index()


ylim = [-1,1]
ylabel = "effect size"
xlabel = "param value norm."
g = sns.relplot(data = df, x = "xnorm", y = "log2FC", hue = "readout", col = "name", row = "pname", kind = "line", height = 2.1)
g.set(ylim = ylim, ylabel = ylabel, xlabel = xlabel)
sns.despine(top = False, right = False)
g.set_titles("{col_name}")
plt.show()

g.savefig("../figures/supplements/parameterscan_modelunspecific.svg")
g.savefig("../figures/supplements/parameterscan_modelunspecific.pdf")
