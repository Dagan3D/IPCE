import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import os

def p_diode(x):
  if x <= 500:
    y = 6.6763288698E-12*x**5 - 1.3543608284E-08*x**4 + 1.0888317706E-05*x**3 - 4.3335323943E-03*x**2 + 8.5399725862E-01*x - 6.6558844853E+01
  elif 500 <= x < 950:
    y = 5.5585093168E-04*x - 2.0552365026E-02
  elif 950 <= x <= 1100:
    y = 3.1698368227E-09*x**4 - 1.2731281453E-05*x**3 + 1.9132245041E-02*x**2 - 1.2752168975E+01*x + 3.1818412914E+03
  return y


def read_file(uploaded_file) -> pd.DataFrame:
  try:
    df = pd.read_table(uploaded_file, encoding='cp1251', skiprows=range(25), sep="  ")
    df = df.drop(["Unnamed: 2", "Unnamed: 4"], axis=1)
    sample = os.path.splitext(uploaded_file.name)[0]
    sample_time = sample+"_Time"
    sample_potential = sample+"_Potential"
    sample_current = sample+"_Current"
    df.columns = [0, sample_time, sample_potential, sample_current]
    df = df.drop([0], axis=1).dropna().copy(deep=True)
    return df
  except:
    st.warning(f'Неверный формат файла "{uploaded_file.name}"', icon="⚠️")
    return None
  
"""## Обработка данных со спектрофотометра для получения зависимости IPCE от длины волны"""
#%% Калибровочный график

with st.expander("Загрузка файлов"):
  
  data_valid = False
  uploaded_files = st.file_uploader("Файлы данных", type = ['edf', 'csv'], accept_multiple_files=True)
  
  currents_sample = pd.DataFrame()
  samples = []
  lens = {}
  for uploaded_file in uploaded_files:
    dataframe = read_file(uploaded_file)
    if dataframe is not None:
      samples.append(os.path.splitext(uploaded_file.name)[0])
      lens[samples[-1]] = len(dataframe)
      lens = pd.Series(lens)
      max_key = lens.idxmax()
      print(max_key) 
      currents_sample = pd.concat([currents_sample, dataframe], axis=1)
      data_valid = True

  if(data_valid):
    lin_sam = [sam+"_Current" for sam in samples]
    print(lin_sam)
    fig = px.line(currents_sample, x=max_key + "_Time", y=lin_sam, labels={'value':"Current"})
    fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    if st.checkbox('Показать таблицу исходных фототоков'):
      currents_sample

      @st.cache_data
      def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(sep=";", index=False).encode('cp1251')

      csv = convert_df(currents_sample)
      
    st.download_button(
      label="Скачать результат",
      data=csv,
      file_name='IPCE.csv',
      mime='text/csv',
      type="primary"
    )

