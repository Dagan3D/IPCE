import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import os

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
  

"### Эта страница поможет вам собрать множество файлов с фототоками в один, а так же учесть площади освещаемых образцов"
with st.expander("Файлы данных"):
    """## Файлы данных.   
    Тут можно загружать сразу несколько файлов.  
    Образцам будет присвоено имя файла.
    """

    data_valid = False
    uploaded_files = st.file_uploader("Файлы данных", type = ['txt', 'csv'], accept_multiple_files=True)
    currents_sample = pd.DataFrame()
    
    samples = []
    for uploaded_file in uploaded_files:
        dataframe = read_file(uploaded_file)
        if dataframe is not None:
            samples.append(os.path.splitext(uploaded_file.name)[0])
            currents_sample['Длина волны, нм'] = dataframe["Длина волны, нм"]
            currents_sample[samples[-1]] = dataframe["Сила тока, мкА"]
            data_valid = True

    if(data_valid):
        fig = px.line(currents_sample.dropna(), x="Длина волны, нм", y=samples, labels={'value':"Сила тока, мкА"})
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        if st.checkbox('Показать таблицу исходных фототоков'):
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