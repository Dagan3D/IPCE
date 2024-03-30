import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from io import StringIO
import os

def p_diod(x):
  if x <= 500:
    y = 6.6763288698E-12*x**5 - 1.3543608284E-08*x**4 + 1.0888317706E-05*x**3 - 4.3335323943E-03*x**2 + 8.5399725862E-01*x - 6.6558844853E+01
  elif 500 <= x < 950:
    y = 5.5585093168E-04*x - 2.0552365026E-02
  elif 950 <= x <= 1100:
    y = 3.1698368227E-09*x**4 - 1.2731281453E-05*x**3 + 1.9132245041E-02*x**2 - 1.2752168975E+01*x + 3.1818412914E+03
  return y


def read_file(uploaded_file) -> pd.DataFrame:
  try:
    if os.path.splitext(uploaded_file.name)[1] == ".txt":
      dataframe = pd.read_table(uploaded_file)
    elif os.path.splitext(uploaded_file.name)[1] == ".csv":
      dataframe = pd.read_csv(uploaded_file, sep=";")
  

    df = pd.DataFrame()
    df["Длина волны, нм"] = dataframe[dataframe.columns[0]]
    df["Сила тока, мкА"] = dataframe[dataframe.columns[1]] * 1E6
    return df
  except:
    st.warning(f'Неверный формат файла "{uploaded_file.name}"', icon="⚠️")
    return None
  

"""## Обработка данных со спектрофотометра для получения зависимости IPCE от длины волны"""
#%% Калибровочный график

with st.expander("Калибровочный график"):
  
  st.markdown('''
          ## Калибровочный график   
          Загрузите его в файле следующего вида:
          В первой колонке должны быть длинны волн (в нанометрах), которыми освещали образец.  
          Во второй колонке - измеренный фототок в амперах.  

          |  Wavelength, nm   |  Current, A   |
          |  ---              | ---           |
          |  300              | 1.2E-3        |
          |  305              | 0.2E-2        | 
          |  ...              | ---           |
          |  600              | 10.0          |   
          ''')
  
  calibration_valid = False
  uploaded_file = st.file_uploader("Файл калибровки", type = ['txt', 'csv'])

  if uploaded_file is not None:
    df = read_file(uploaded_file)
    if df is not None:
      df["K_diod"] = df['Длина волны, нм'].apply(p_diod)
      df["Мощность излучения, мкВт"] = df["Сила тока, мкА"]/df["K_diod"]
      calibration_valid = True

      fig = px.line(df, x="Длина волны, нм", y="Мощность излучения, мкВт")
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      if st.checkbox('Показать таблицу исходных данных калибровки'):
        df

#%%Файлы данных
if calibration_valid:
  with st.expander("Файлы данных"):
    "## Файлы данных"

    data_valid = False
    uploaded_files = st.file_uploader("Файлы данных", type = ['txt', 'csv'], accept_multiple_files=True)
    
    currents_sample = pd.DataFrame()
    currents_sample['Длина волны, нм'] = df["Длина волны, нм"]
    samples = []
    for uploaded_file in uploaded_files:
      dataframe = read_file(uploaded_file)
      if dataframe is not None:
        samples.append(os.path.splitext(uploaded_file.name)[0])
        currents_sample[samples[-1]] = dataframe["Сила тока, мкА"]
        data_valid = True

    if(data_valid):
      fig = px.line(currents_sample.dropna(), x="Длина волны, нм", y=samples, labels={'value':"Сила тока, мкА"})
      fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      if st.checkbox('Показать таблицу исходных данных фототоков'):
        currents_sample

#%% Учёт площади образца
  if(data_valid):
    with st.expander("Учёт площади образца"):
      "## Пересчёт фототока в плотность фототока для учёта различной площади образца"
      st.markdown(r'''$I_{удельный} =\frac{I_{образца}}{S_{образца}}$''')
      area_sample = pd.DataFrame()
      area_sample["Образец"] = samples
      area_sample["Площадь образца, см^2"] = 7.0
      area_sample = area_sample.astype({"Площадь образца, см^2" : float})
      area_sample = st.data_editor(area_sample)
      

      density_current = pd.DataFrame()
      density_current['Длина волны, нм'] = currents_sample['Длина волны, нм']
      
      for current_sample in samples:
        area = area_sample.loc[area_sample["Образец"] == current_sample]["Площадь образца, см^2"]
        density_current[current_sample] = currents_sample[current_sample]/(float(area))
        
      fig = px.line(density_current.dropna(), x="Длина волны, нм", y=samples, labels={'value': r"Плотность тока, мкА/см^2"})
      fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      if st.checkbox('Показать таблицу плотности фототоков'):
        density_current

#%% Пересчёт в IPCE
  if(data_valid):
    with st.expander("Пересчёт в IPCE"):
      "## Пересчёт из плотности тока в IPCE"
      st.markdown(r'''Тут производится пересчёт величины тока в эффективность преобразования фотона в электрон, по формуле:  
                  $IPCE =I_{удельный}~*~ \frac{1240}{\lambda}~ / ~ P_{падающая}$''')
      IPCE_sample = pd.DataFrame()
      IPCE_sample["Длина волны, нм"] = density_current['Длина волны, нм']
      for current_sample in samples:
        IPCE_sample[current_sample] = (density_current[current_sample]/1E6*10000)*1240/currents_sample["Длина волны, нм"]/df["Мощность излучения, мкВт"]*100
      
      fig = px.line(IPCE_sample.dropna(), x="Длина волны, нм", y=samples, labels={'value':"IPCE, %"})
      fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      if st.checkbox('Показать таблицу полученных данных IPCE'):
        IPCE_sample
      

