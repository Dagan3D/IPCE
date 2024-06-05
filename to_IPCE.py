import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from statsmodels.tsa.seasonal import STL

def p_diode(x):
  if x <= 500:
    y = 6.6763288698E-12*x**5 - 1.3543608284E-08*x**4 + 1.0888317706E-05*x**3 - 4.3335323943E-03*x**2 + 8.5399725862E-01*x - 6.6558844853E+01
  elif 500 <= x < 950:
    y = 5.5585093168E-04*x - 2.0552365026E-02
  elif 950 <= x <= 1100:
    y = 3.1698368227E-09*x**4 - 1.2731281453E-05*x**3 + 1.9132245041E-02*x**2 - 1.2752168975E+01*x + 3.1818412914E+03
  return y

@st.cache_data  
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_table(path,  decimal=',')
    format_data = "%m%d%y_%H%M%S"

    df["Time"] = df.Timestamp.apply( lambda x: datetime.strptime(x, format_data))
    df["Time"] = df["Time"] - df["Time"][0]
    df["Time"] = df["Time"].dt.total_seconds()
    df["Shutter"] = df.Shutter.shift(-23)
    return df

@st.cache_data
def reduction_smooth(df: pd.DataFrame, window: int = 10):
    rolling_mean = df['Current'].ewm(alpha=0.2, adjust=False).mean()
    df_resampled = pd.DataFrame({'Time': df['Time'].iloc[::window],
                                 'Current': rolling_mean[::window],
                                 'Wavelength': df['Wavelength'].iloc[::window],
                                 "Shutter":  df['Shutter'].iloc[::window]})
    df = df.reset_index(drop=True)
    return df_resampled

@st.cache_data
def time_split(df_: pd.DataFrame, start_wave: int = 300, stop_wave: int = 1100) -> pd.DataFrame:
    df = df_.copy(deep=True)
    for i in range(int(df.Time[df.index[-1]])):
        pattern = np.linspace(0,1,len(df[df.Time == i]))
        df.loc[df.Time == i, "Time"] += pattern
    df = df.loc[df.Wavelength >= start_wave][["Time", "Current", "Wavelength", "Shutter"]].copy(deep=True)
    df = df.reset_index(drop=True)
    return df

@st.cache_data
def cut_baseline(df: pd.DataFrame, window: int = 10, measure_in_monowave: int = 800) -> pd.DataFrame:
    df_ = df.copy(deep=True)
    stl = STL(df_.Current.dropna(), period=measure_in_monowave)
    res = stl.fit()
    df_["Photocurrent"] = res.seasonal
    return df_

@st.cache_data
def get_photocurrent(df: pd.DataFrame, window:int = 10, left_shift: int = 10, right_shift: int = 5) -> pd.DataFrame:
    IPCE_df = pd.DataFrame([])
    df = df.copy(deep=True)
    df["Current"] = df["Photocurrent"].rolling(window=window).median()
    for wavelength in df["Wavelength"].unique():
        mono_df = df.loc[df["Wavelength"] == wavelength][left_shift:-1*right_shift]
        photocurrent = mono_df.Current.max() - mono_df.Current.min()
        new_row = pd.DataFrame({"Wavelength": wavelength, "Photocurrent": photocurrent}, index=[len(IPCE_df)])
        IPCE_df = pd.concat([IPCE_df, new_row])
    return IPCE_df

def mean_measure(df):
  lens = []
  for wavelength in df["Wavelength"].unique():
    lens.append(len(df.loc[df.Wavelength == wavelength]))
  return sum(lens)/len(lens)
   

def plot_graf():
    pass