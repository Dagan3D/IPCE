#%%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv(r'C:\Users\butma\Documents\Python\IPCE\Данные\L1 10-6.sf')
df
# %%
dataframe = pd.read_table(r'C:\Users\butma\Documents\Python\IPCE\Данные\L1 10-6.sf',
                          encoding="cp1251", sep="             ", engine="python",
                        skiprows=17, decimal='.').dropna()
df = pd.DataFrame()
dataframe = dataframe.reset_index(drop=False)
df["Длина волны, нм"] = dataframe["index"].convert_dtypes()
df["Интенсивность"] = dataframe[dataframe.columns[1]]
df = df.astype(float)
# %%
df
# %%
