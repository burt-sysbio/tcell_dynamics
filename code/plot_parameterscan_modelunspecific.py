from exp import Simulation, SimList
import models as model
from params_fig2 import d
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("paper_theme_python.mplstyle")
sns.set_palette("deep")

out_il2 = pd.read_csv("../output/paramscans/pscan_il2_unspecific.csv")
out1 = pd.read_csv("../output/paramscans/pscan_timer_unspecific.csv")
out2 = pd.read_csv("../output/paramscans/pscan_carry_unspecific.csv")

df = pd.concat([out_il2, out1, out2]).reset_index()


ylim = [-1,1]
ylabel = "effect size"
xlabel = "param value norm."
g = sns.relplot(data = df, x = "xnorm", y = "log2FC", hue = "readout", row = "name", col = "pname", kind = "line", height = 1.5)
g.set(ylim = ylim, ylabel = ylabel, xlabel = xlabel)
sns.despine(top = False, right = False)
g.set_titles("{col_name}")
plt.show()

g.savefig("../figures/supplements/parameterscan_modelunspecific.svg")
g.savefig("../figures/supplements/parameterscan_modelunspecific.pdf")
