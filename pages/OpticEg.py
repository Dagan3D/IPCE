import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import os

def read_file(uploaded_file) -> pd.DataFrame:

    if os.path.splitext(uploaded_file.name)[1] == ".txt":
        dataframe = pd.read_table(uploaded_file)
        df = pd.DataFrame()
        df["Длина волны, нм"] = dataframe[dataframe.columns[0]]
        df["Интенсивность"] = dataframe[dataframe.columns[1]] 
    elif os.path.splitext(uploaded_file.name)[1] == ".csv":
        dataframe = pd.read_csv(uploaded_file, sep=";")
        df = pd.DataFrame()
        df["Длина волны, нм"] = dataframe[dataframe.columns[0]]
        df["Интенсивность"] = dataframe[dataframe.columns[1]] 
    elif os.path.splitext(uploaded_file.name)[1] == ".xls":
        st.write(uploaded_file.name)
        dataframe = pd.ExcelFile(uploaded_file).parse("Лист1")
        #st.write(dataframe.columns)
        df = pd.DataFrame()
        df["Длина волны, нм"] = dataframe[dataframe.columns[12]]
        df["Интенсивность"] = dataframe[dataframe.columns[13]]       
    else:
        raise Exception()
    return df

  

"### Эта страница поможет вам собрать множество файлов c графиком оптического поглощения в один и пересчитать полученные значения в оптическую ширину запрещенной зоны."
with st.expander("Файлы данных"):
    """## Файлы данных.   
    Тут можно загружать сразу несколько файлов.  
    Образцам будет присвоено имя файла.
    """

    data_valid = False
    uploaded_files = st.file_uploader("Файлы данных", type = ['txt', 'csv', 'xls'], accept_multiple_files=True)
    currents_sample = pd.DataFrame()
    
    samples = []
    for uploaded_file in uploaded_files:
        dataframe = read_file(uploaded_file)
        if dataframe is not None:
            samples.append(os.path.splitext(uploaded_file.name)[0])
            currents_sample['Длина волны, нм'] = dataframe["Длина волны, нм"]
            currents_sample[samples[-1]] = dataframe["Интенсивность"]
            data_valid = True

    if(data_valid):
        fig = px.line(currents_sample.dropna(), x="Длина волны, нм", y=samples, labels={'value':"Отражение, %"})
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        if st.checkbox('Показать таблицу исходных данных'):
            currents_sample

#%% Учёт площади образца
if(data_valid):
    with st.expander("Учёт площади образца"):

        OpticEg = pd.DataFrame()
        OpticEg["hv"] = (1240 / (currents_sample["Длина волны, нм"]))

        #"## Пересчёт фототока в плотность фототока для учёта различной площади образца"
        #st.markdown(r'''$I_{удельный} =\frac{I_{образца}}{S_{образца}}$''')

        degree = st.selectbox('Какую степень использовать при вычислениях', (2, 1/2))
        
        for current_sample in samples:
            df_cub_munk = pd.DataFrame()
            df_cub_munk['hv'] = (1240 / (currents_sample['Длина волны, нм']))
            df_cub_munk['R'] = currents_sample[current_sample]
            df_cub_munk['R%'] = df_cub_munk['R'] / 100
            df_cub_munk['(1-R)^2'] = (1 - df_cub_munk['R%']) ** 2
            df_cub_munk['2R'] = df_cub_munk['R%'] * 2
            df_cub_munk['k/s'] = df_cub_munk['(1-R)^2'] / df_cub_munk['2R']
            df_cub_munk['F*hv'] = df_cub_munk['k/s'] * df_cub_munk['hv']
            df_cub_munk[current_sample] = df_cub_munk['F*hv'] ** degree
            name = os.path.basename(current_sample)
            name = os.path.splitext(name)[0]
            OpticEg[current_sample] = df_cub_munk[current_sample]
            
        fig = px.line(OpticEg.dropna(), x="hv", y=samples, labels={'value': f"a*h*nu ^ {degree}"})
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
        
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        if st.checkbox('Показать таблицу результатов'):
            OpticEg

        @st.cache_data
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(sep=";", index=False).encode('cp1251')

        csv = convert_df(OpticEg)

        st.download_button(
            label="Скачать результат",
            data=csv,
            file_name='OpticEg.csv',
            mime='text/csv',
            type="primary"
        )
