import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import os
import to_IPCE

"""## Обработка данных со спектрофотометра для получения зависимости IPCE от длины волны"""
#%% Калибровочный график

with st.expander("Калибровочный график"):
  
  st.markdown('''
    Этот обработчик был испытан со следующими параметрами на приборе:   
              Длинны волн: 280 - 450 нм  
              Шаг длинны волны: 5 нм  
              Интервал меду измерениями: 50 мс  
              Время изменения в темноте и на свету: 30 секунд  
              Время ожидания до измерения: 120 секунд  
              Смещение на образце: 200 мВ  
              Чувствительность: 1 мкА  
          ''')
  
  calibration_valid = False
  uploaded_file = st.file_uploader("Файл калибровки", type = ['txt', 'csv'])
  

  if uploaded_file is not None:
    df = to_IPCE.read_data(uploaded_file)
    df = to_IPCE.reduction_smooth(df, window=1)
    df = to_IPCE.time_split(df)
    df["Photocurrent"] = df["Current"]
    df = to_IPCE.get_photocurrent(df, window=10) 
    if df is not None:
      df["K_diode"] = df['Wavelength'].apply(to_IPCE.p_diode)
      df["Мощность излучения, мкВт"] = df["Photocurrent"]/df["K_diode"]
      calibration_valid = True

      fig = px.line(df, x="Wavelength", y="Мощность излучения, мкВт")
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      if st.checkbox('Показать таблицу данных калибровки'):
        df

#%%Файлы данных
if calibration_valid:
  with st.expander("Файлы данных"):
    """## Файлы данных.   
    Тут можно загружать сразу несколько файлов.  
    Образцам будет присвоено имя файла.
    """

    data_valid = False
    uploaded_files = st.file_uploader("Файлы данных", type = ['txt', 'csv'], accept_multiple_files=True)
    
    
    df_photocurrent = pd.DataFrame()
    samples = []
    for uploaded_file in uploaded_files:
            
      dataframe = to_IPCE.read_data(uploaded_file)
      dataframe = to_IPCE.time_split(dataframe)
      
      if st.checkbox('Показать таблицу исходный график'):
        df_photocurrent 
        fig = px.line(dataframe.dropna(), x="Time", y="Current", labels={'value':"Сила тока, мкА"})
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        dataframe

      
      dataframe = to_IPCE.reduction_smooth(dataframe)
      
      if dataframe is not None:
        samples.append(os.path.splitext(uploaded_file.name)[0])
        if ("Time" not in df_photocurrent.columns) or (len(df_photocurrent["Time"] ) < len(dataframe.Time)):
          df_photocurrent["Time"] = dataframe.Time
        df_photocurrent[samples[-1]] = dataframe["Current"]
        data_valid = True    

    if data_valid:
      fig = px.line(df_photocurrent.dropna(), x="Time", y=samples, labels={'value':"Сила тока, мкА"})
      fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      if st.checkbox('Показать таблицу исходных фототоков'):
        df_photocurrent    

      

#%% Выгрузка фототоков
  if data_valid:
    with st.expander("Извлечение фототоков"):
      "## Извлечение фототоков из полученных данных"

      measure = to_IPCE.mean_measure(dataframe)
      measure_in_monowave = st.number_input("Insert a number", step = 1, format="%i", value = round(measure))
      
      currents_sample = pd.DataFrame()
      currents_sample['Длина волны, нм'] = df["Wavelength"]
      
      dataframe = to_IPCE.cut_baseline(dataframe, measure_in_monowave=measure_in_monowave)
      fig = px.line(dataframe.dropna(), x="Time", y="Photocurrent", labels={'value':"Сила тока, мкА"})
      fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      if st.checkbox('Показать таблицу фототоков без базовой линии'):
        dataframe

      dataframe = to_IPCE.get_photocurrent(dataframe, window=10)
      if dataframe is not None:
        currents_sample[samples[-1]] = dataframe["Photocurrent"]
        
        
        fig = px.line(currents_sample.dropna(), x="Длина волны, нм", y=samples, labels={'value':"Сила тока, мкА"})
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        if st.checkbox('Показать таблицу фототоков'):
          currents_sample
  

#%% Учёт площади образца
  if data_valid:
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
        IPCE_sample[current_sample] = (density_current[current_sample])*1240/currents_sample["Длина волны, нм"]/df["Мощность излучения, мкВт"]*100
      
      fig = px.line(IPCE_sample.dropna(), x="Длина волны, нм", y=samples, labels={'value':"IPCE, %"})
      fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      if st.checkbox('Показать таблицу IPCE'):
        IPCE_sample

      @st.cache_data
      def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(sep=";", index=False).encode('cp1251')

      csv = convert_df(IPCE_sample)
      
      st.download_button(
        label="Скачать результат",
        data=csv,
        file_name='IPCE.csv',
        mime='text/csv',
        type="primary"
      )
