import streamlit as st
import pandas as pd
import numpy as np

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
  


"""Это программа для обработки данных фототока полученных на спектрофотометре.   
Приложение поддерживает следующий файлы формата .txt и .csv:    
"""


with st.expander("Калибровочный график"):
  
  st.markdown('''
          ## Калибровочный график   
          Загрузите его в формате таблицы со столбцами:   
          **Wavelength, nm - Plot 0** и **I, A - Plot 0**.''')
  
  calibration_valid = False
  uploaded_file = st.file_uploader("Файл калибровки")

  if uploaded_file is not None:
    df = read_file(uploaded_file)
    if df is not None:
      df["K_diod"] = df['Длина волны, нм'].apply(p_diod)
      df["Мощность излучения, мкВт"] = df["Сила тока, мкА"]/df["K_diod"]
      st.line_chart(df, x="Длина волны, нм", y = [ "Мощность излучения, мкВт"])
      calibration_valid = True
      if st.checkbox('Показать таблицу исходных данных калибровки'):
        df

if calibration_valid:
  with st.expander("Файлы данных"):
    "## Файлы данных"
    "Загрузите их в формате: Wavelength, nm - Plot 0	I, A - Plot 0"
    "Полученному образцу будет присвоено имя файла"

    data_valid = False
    uploaded_files = st.file_uploader("Файлы данных", accept_multiple_files=True)
    currents_sample = pd.DataFrame()
    currents_sample['Длина волны, нм'] = df["Длина волны, нм"]
    for uploaded_file in uploaded_files:
      dataframe = read_file(uploaded_file)
      if dataframe is not None:
        currents_sample[os.path.splitext(uploaded_file.name)[0]] = dataframe["Сила тока, мкА"]
        data_valid = True
    
    if(data_valid):
      currents_sample.dropna()
      st.line_chart(currents_sample.dropna(), x='Длина волны, нм')
      if st.checkbox('Показать таблицу исходных данных фототоков'):
        currents_sample

  if(data_valid):
    with st.expander("Пересчёт в IPCE"):
      "## Пересчёт из тока и площади в IPCE"
      st.markdown(r'''Тут производится пересчёт величины тока в эффективность преобразования фотона в электрон, по формуле:  
                  $IPCE =\frac{I_{образца}}{S_{образца}}~*~ \frac{1240}{\lambda}~ / ~ P_{падающая}$''')
      IPCE_sample = pd.DataFrame()

      samples = currents_sample.drop("Длина волны, нм", axis=1).columns

      default_area_sample = pd.DataFrame()
      default_area_sample["Образец"] = samples
      default_area_sample["Площадь образца, см2"] = 7
      area_sample = st.data_editor(default_area_sample)

      IPCE_sample["Длина волны, нм"] = currents_sample['Длина волны, нм']
      for current_sample in samples:
        area = area_sample.loc[area_sample["Образец"] == current_sample]["Площадь образца, см2"]
        IPCE_sample[current_sample] = (currents_sample[current_sample]/1E6/(int(area)/10000))*1240/currents_sample["Длина волны, нм"]/df["Мощность излучения, мкВт"]*100
      
      st.line_chart(IPCE_sample.dropna(), x='Длина волны, нм')
      if st.checkbox('Показать таблицу полученных данных IPCE'):
        IPCE_sample
      

