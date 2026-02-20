import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import os
from scipy import interpolate


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
        dataframe = pd.read_table(uploaded_file, encoding="cp1251",
                                  sep="  ", engine="python", skiprows=14, decimal='.').dropna()
        df = pd.DataFrame()
        dataframe = dataframe.reset_index(drop=False)
        df["Длина волны, нм"] = dataframe["index"].convert_dtypes()
        df["Интенсивность"] = dataframe[dataframe.columns[1]]
        df = df.astype(float)
    elif os.path.splitext(uploaded_file.name)[1] == ".sf":
        dataframe = pd.read_table(uploaded_file,
                                  encoding="cp1251", header=None, sep=r"\s+", engine="python",
                                  skiprows=18, decimal='.', skipfooter=4).dropna()
        df = pd.DataFrame()
        print(dataframe)
        dataframe = dataframe.reset_index(drop=False)
        df["Длина волны, нм"] = dataframe[dataframe.columns[1]].convert_dtypes()
        df["Интенсивность"] = dataframe[dataframe.columns[2]]
        df = df.astype(float)
    else:
        raise Exception()
    return df


def data_correction(df, correction_list=[339, 340, 387, 388, 389, 390, 453, 565], smooth=4):
    """Исправление скачка на спектрофотометре"""
    nm = df["Длина волны, нм"]
    for corr in correction_list:
        for col in df.columns:
            # Проверяем, что точки коррекции есть в данных
            if corr in nm.values and (corr + 1) in nm.values:
                diff_val = df.loc[nm == corr+1, col].values - df.loc[nm == corr, col].values
                if len(diff_val) > 0:
                    df.loc[nm > corr, col] = df.loc[nm > corr, col] - diff_val[0]
            if smooth > 0:
                df[col] = df[col].ewm(smooth).mean().dropna()
    return df


"### Эта страница поможет вам собрать множество файлов c графиком оптического поглощения в один и пересчитать полученные значения в оптическую ширину запрещенной зоны."

# Выбор режима загрузки
load_mode = st.radio(
    "Режим загрузки данных",
    ["Спектры (%)", "Интенсивности"],
    horizontal=True,
    help="Выберите 'Спектры (%)' если у вас готовые спектры отражения/пропускания в процентах. \
          Выберите 'Интенсивности' если у вас сырые данные интенсивностей с датчика."
)

data_valid = False
currents_sample = pd.DataFrame()
samples = []
df_reference = None  # Для хранения интенсивностей эталона

if load_mode == "Спектры (%)":
    # Текущий режим - загрузка готовых спектров
    with st.expander("Файлы данных"):
        """## Файлы данных.   
        Тут можно загружать сразу несколько файлов.  
        Образцам будет присвоено имя файла.
        """

        uploaded_files = st.file_uploader("Файлы данных", type=[
                                          'txt', 'csv', 'xls', 'pts', 'sf'], accept_multiple_files=True)
        
        smooth = st.checkbox("Сглаживание")
        if smooth:
            smooth_force = st.slider("Сила сглаживания", 0., 2., 1., 0.01)
        else:
            smooth_force = 0.0

        for uploaded_file in uploaded_files:
            dataframe = read_file(uploaded_file)
            if smooth:
                dataframe = data_correction(dataframe, smooth=int(smooth_force))
            if dataframe is not None:
                samples.append(os.path.splitext(uploaded_file.name)[0])
                currents_sample['Длина волны, нм'] = dataframe["Длина волны, нм"]
                currents_sample[samples[-1]] = dataframe["Интенсивность"]
                data_valid = True

        if (data_valid):
            fig = px.line(currents_sample.dropna(), x="Длина волны, нм",
                          y=samples, labels={'value': "Отражение, %"})
            fig.update_layout(legend=dict(yanchor="top", xanchor="right"))
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            if st.checkbox('Показать таблицу исходных данных'):
                currents_sample

else:
    # Новый режим - работа с интенсивностями
    with st.expander("Интенсивности эталона (I₀)"):
        """## Интенсивности эталона (I₀)
        Загрузите файл с интенсивностями эталона (чистая подложка для отражения, 
        или воздух/растворитель для пропускания).
        """
        
        reference_file = st.file_uploader(
            "Файл эталона (I₀)", 
            type=['txt', 'csv', 'xls', 'pts', 'sf'],
            key="reference_uploader"
        )
        
        if reference_file is not None:
            df_reference = read_file(reference_file)
            df_reference = df_reference.rename(columns={"Интенсивность": "I₀"})
            st.success(f"Загружен эталон: {reference_file.name}")
            
            # Визуализация эталона
            fig_ref = px.line(df_reference, x="Длина волны, нм", y="I₀",
                             labels={'I₀': "Интенсивность эталона, I₀"})
            st.plotly_chart(fig_ref, theme="streamlit", use_container_width=True)

    with st.expander("Интенсивности образцов (I)"):
        """## Интенсивности образцов (I)
        Загрузите файлы с интенсивностями измеряемых структур.
        Коэффициент отражения/пропускания будет рассчитан как R = I/I₀.
        """
        
        uploaded_files = st.file_uploader(
            "Файлы образцов (I)", 
            type=['txt', 'csv', 'xls', 'pts', 'sf'], 
            accept_multiple_files=True,
            key="samples_uploader"
        )
        
        smooth_int = st.checkbox("Сглаживание", key="smooth_intensities")
        if smooth_int:
            smooth_force = st.slider("Сила сглаживания", 0., 2., 1., 0.01, key="smooth_force_int")
        else:
            smooth_force = 0.0

        if df_reference is not None and uploaded_files:
            # Объединяем данные эталона с образцами
            wavelengths = df_reference["Длина волны, нм"].values
            
            for uploaded_file in uploaded_files:
                df_sample = read_file(uploaded_file)
                
                if smooth_int:
                    df_sample = data_correction(df_sample, smooth=int(smooth_force))
                
                if df_sample is not None:
                    sample_name = os.path.splitext(uploaded_file.name)[0]
                    samples.append(sample_name)
                    
                    # Интерполяция интенсивностей образца на длины волн эталона
                    # (на случай если сетки длин волн немного отличаются)
                    
                    interp_func = interpolate.interp1d(
                        df_sample["Длина волны, нм"].values,
                        df_sample["Интенсивность"].values,
                        kind='linear',
                        fill_value='extrapolate'  # type: ignore
                    )
                    I_sample = interp_func(wavelengths)
                    
                    # Расчёт R = I/I₀
                    I_reference = df_reference["I₀"].values
                    R_relative = I_sample / I_reference
                    
                    currents_sample['Длина волны, нм'] = wavelengths
                    currents_sample[sample_name] = R_relative
            
            data_valid = True
            
            # Визуализация интенсивностей эталона и образцов на одном графике
            st.subheader("Интенсивности эталона и образцов")
            intensity_df = currents_sample.copy()
            intensity_df['I₀ (эталон)'] = df_reference["I₀"].values
            cols_to_plot = ['I₀ (эталон)'] + samples
            fig_int = px.line(intensity_df.dropna(), x="Длина волны, нм",
                             y=cols_to_plot, labels={'value': "Интенсивность"})
            fig_int.update_layout(legend=dict(yanchor="top", xanchor="right"))
            st.plotly_chart(fig_int, theme="streamlit", use_container_width=True)
            
            # Визуализация рассчитанного R
            st.subheader("Рассчитанный коэффициент (R = I/I₀)")
            fig_R = px.line(currents_sample.dropna(), x="Длина волны, нм",
                           y=samples, labels={'value': "R (отн. ед.)"})
            fig_R.update_layout(legend=dict(yanchor="top", xanchor="right"))
            st.plotly_chart(fig_R, theme="streamlit", use_container_width=True)
            
            if st.checkbox('Показать таблицу рассчитанных коэффициентов'):
                currents_sample

# %% Учёт площади образца
if (data_valid):
    with st.expander("Учёт площади образца"):

        OpticEg = pd.DataFrame()
        print(currents_sample['Длина волны, нм'])
        OpticEg["Длина волны, нм"] = currents_sample['Длина волны, нм']
        OpticEg["hv"] = (1240 / (currents_sample["Длина волны, нм"]))

        degree = st.selectbox(
            'Какую степень использовать при вычислениях', (2, 1/2))
        x_asxi = st.selectbox("Подписи горизонтальной оси",
                              ('hv', 'Длина волны, нм'))

        for current_sample in samples:
            df_cub_munk = pd.DataFrame()
            df_cub_munk['hv'] = (1240 / (currents_sample['Длина волны, нм']))
            df_cub_munk['R'] = currents_sample[current_sample]
            
            # Для режима интенсивностей R уже в относительных единицах
            # Для режима спектров R в процентах
            if load_mode == "Спектры (%)":
                df_cub_munk['R_frac'] = df_cub_munk['R'] / 100
            else:
                # R уже в относительных единицах (0-1), не нужно делить на 100
                df_cub_munk['R_frac'] = df_cub_munk['R']
            
            df_cub_munk['(1-R)^2'] = (1 - df_cub_munk['R_frac']) ** 2
            df_cub_munk['2R'] = df_cub_munk['R_frac'] * 2
            df_cub_munk['k/s'] = df_cub_munk['(1-R)^2'] / df_cub_munk['2R']
            df_cub_munk['F*hv'] = df_cub_munk['k/s'] * df_cub_munk['hv']
            df_cub_munk[current_sample] = df_cub_munk['F*hv'] ** degree
            name = os.path.basename(current_sample)
            name = os.path.splitext(name)[0]
            OpticEg[current_sample] = df_cub_munk[current_sample]

        fig = px.line(OpticEg.dropna(), x=x_asxi, y=samples,
                      labels={'value': f"a*h*nu ^ {degree}"})
        fig.update_layout(legend=dict(yanchor="top", xanchor="right"))

        if x_asxi == 'hv':
            fig.update_layout(
                xaxis2=dict(
                    title="Длина волны, нм",
                    overlaying="x",
                    side="top",
                    # Преобразование диапазона энергии в длину волны
                    range=[1240 / 20, 1240 / 0.1],
                    # Пример значений для оси
                    tickvals=[1240 / x for x in [20, 10, 5, 2, 1]],
                    ticktext=[str(x)
                              for x in [20, 10, 5, 2, 1]],  # Подписи для оси
                ),
                legend=dict(yanchor="top", xanchor="right")
            )
        else:
            fig.update_layout(xaxis=dict(autorange="reversed"))

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

# %%