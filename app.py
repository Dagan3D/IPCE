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

"## Это помощник в обработке файлов для IPCE снятых в 4205 и сохранённых в файлах .txt или .csv"
with st.expander("Калибровочный график"):
  st.markdown('''
          ## Калибровочный график   
          Загрузите его в формате таблицы со столбцами:   
          **Wavelength, nm - Plot 0** и **I, A - Plot 0**.''')
  calibration_valid = False
  uploaded_file = st.file_uploader("Файл калибровки")
  if uploaded_file is not None:
    dataframe = pd.read_table(uploaded_file)
    df = pd.DataFrame()
    df["Длина волны, нм"] = dataframe["Wavelength, nm - Plot 0"]
    df["Cила тока, мкА"] = dataframe["I, A - Plot 0"] * 1E6
    df["K_diod"] = df['Длина волны, нм'].apply(p_diod)
    df["Мощность излучения, мкВт"] = dataframe["I, A - Plot 0"]/df["K_diod"]*10e6
    st.line_chart(df, x="Длина волны, нм", y = [ "Мощность излучения, мкВт"])
    calibration_valid = True

if calibration_valid:
  with st.expander("Файлы данных"):
    "## Файлы данных"
    "Загрузите их в формате: Wavelength, nm - Plot 0	I, A - Plot 0"
    "Полученному образцу будет присвоено имя файла"

    uploaded_files = st.file_uploader("Файлы данных [.txt]", accept_multiple_files=True)
    currents_sample = pd.DataFrame()
    currents_sample['Длина волны, нм'] = df["Длина волны, нм"]
    for uploaded_file in uploaded_files:
      dataframe = pd.read_table(uploaded_file)  
      currents_sample[os.path.splitext(uploaded_file.name)[0]] = dataframe["I, A - Plot 0"] * 1E6

    
    if(len(currents_sample.columns) > 1):
      currents_sample.dropna()
      st.line_chart(currents_sample.dropna(), x='Длина волны, нм')
      if st.checkbox('Показать таблицу исходных данных'):
        currents_sample

  if(len(currents_sample.columns) > 1):
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
      if st.checkbox('Показать таблицу полученных данных'):
        IPCE_sample
      

