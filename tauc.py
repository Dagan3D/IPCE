#%%
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import plotly.express as px

plt.style.use("default")

#%%
def tauc_plot(file):
    df = pd.read_csv(file, delimiter="  ", engine="python", header=None, skiprows=17, encoding="cp1251")
    df.columns = ["nm", "%"]
    df = df[0:910].copy()
    df = df.astype(float)
    df["F"] = ((1-df["%"])**2)/(2*df["%"])
    df["hv"] = scipy.constants.h * scipy.constants.c / (df["nm"]/1e9)/scipy.constants.eV
    df["(F*hv)^0.5"] = (df["F"]*df["%"])**0.5
    # print(df)
    return df[["hv", "(F*hv)^0.5"]]

# %%
files = ["SnTiO3_кислота.pts", "SnTiO3_щёлочь.pts", "BaTiO3 отожжённый vs Ti пропускание.pts", "TiO2 vs Ti пропускание.pts"]
df = pd.DataFrame()

df["hv"] = tauc_plot(files[0])["hv"]
for file in files:
    df_tauc = tauc_plot(file)
    df[file] = df_tauc[df_tauc.columns[1]]

df.plot(x="hv")
plt.ylim(0, df[df.columns[-1]].max()+2)
df.to_csv("tauc.csv")    
# %%
