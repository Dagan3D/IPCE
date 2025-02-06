import streamlit as st # type: ignore
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
        dataframe = pd.ExcelFile(uploaded_file).parse("Лист1")
        df = pd.DataFrame()
        df["Длина волны, нм"] = dataframe[dataframe.columns[12]]
        df["Интенсивность"] = dataframe[dataframe.columns[13]]
    elif os.path.splitext(uploaded_file.name)[1] == ".pts":
        dataframe = pd.read_table(uploaded_file, encoding="cp1251", sep="  ", engine="python", skiprows=14, decimal='.').dropna()
        df = pd.DataFrame()
        dataframe = dataframe.reset_index(drop=False)
        df["Длина волны, нм"] = dataframe["index"].convert_dtypes()
        df["Интенсивность"] = dataframe[dataframe.columns[1]]
        df = df.astype(float)        
    else:
        raise Exception()
    return df

def data_correction(df, correction_list=[339, 340, 387, 388, 389, 390, 453, 565], smooth=4):
    """Исправление скачка на спектрофотометре"""
    nm = df["Длина волны, нм"]
    for corr in correction_list:
        for col in df.columns:
            df.loc[nm > corr, col] = df.loc[nm > corr, col] - \
                (df.loc[nm == corr+1, col].values -
                 df.loc[nm == corr, col].values)[0]
        if smooth > 0:
            df[col] = df[col].ewm(smooth).mean().dropna()
    return df

"### Эта страница поможет вам собрать множество файлов c графиком оптического поглощения в один и пересчитать полученные значения в оптическую ширину запрещенной зоны."
with st.expander("Файлы данных"):
    """## Файлы данных.   
    Тут можно загружать сразу несколько файлов.  
    Образцам будет присвоено имя файла.
    """

    data_valid = False
    uploaded_files = st.file_uploader("Файлы данных", type = ['txt', 'csv', 'xls', 'pts'], accept_multiple_files=True)
    currents_sample = pd.DataFrame()
    if smooth := st.checkbox("Сглаживание"):
        smooth_force = st.slider("Сила сглаживания", 0., 2., 1., 0.01)
    

    samples = []
    for uploaded_file in uploaded_files:
        dataframe = read_file(uploaded_file)
        if smooth:
            dataframe = data_correction(dataframe, smooth=smooth_force)
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
        print(currents_sample['Длина волны, нм'])
        OpticEg["Длина волны, нм"] = currents_sample['Длина волны, нм']
        OpticEg["hv"] = (1240 / (currents_sample["Длина волны, нм"]))

        #"## Пересчёт фототока в плотность фототока для учёта различной площади образца"
        #st.markdown(r'''$I_{удельный} =\frac{I_{образца}}{S_{образца}}$''')
        
        degree = st.selectbox('Какую степень использовать при вычислениях', (2, 1/2))
        x_asxi = st.selectbox("Подписи горизонтальной оси", ('hv', 'Длина волны, нм'))

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
            
        fig = px.line(OpticEg.dropna(), x=x_asxi, y=samples, labels={'value': f"a*h*nu ^ {degree}"})
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))


        if x_asxi == 'hv':
            fig.update_layout(
                xaxis2=dict(
                    title="Длина волны, нм",
                    overlaying="x",
                    side="top",
                    range=[1240 / 20, 1240 / 0.1],  # Преобразование диапазона энергии в длину волны
                    tickvals=[1240 / x for x in [20, 10, 5, 2, 1]],  # Пример значений для оси
                    ticktext=[str(x) for x in [20, 10, 5, 2, 1]],  # Подписи для оси
                ),
                legend=dict(yanchor="top", xanchor="right")
            )
        else:
            fig.update_layout(xaxis=dict(autorange="reversed"))
        
        # if x_asxi != 'hv':  
        #     fig.update_layout(xaxis = dict(autorange="reversed"))
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
