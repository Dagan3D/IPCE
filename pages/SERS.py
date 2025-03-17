import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from io import StringIO
import os
import SERS_analis.SERS_find_pick as sers

st.sidebar.header("Параметры обработки")
peaks_points_list = st.sidebar.text_input(
    "Возможные положения пиков", "1170-1188, 1360-1378, 1610-1625").replace(' ', '').split(",")

st.title("SERS analis")
lam = 10**st.sidebar.slider("Параметр сглаживания", 1, 15, 7)
itermax = st.sidebar.slider(
    "Параметр максимального числа итераций сглаживания", 1, 100, 20)

peaks_points = [tuple(x.split("-")) for x in peaks_points_list]
print(peaks_points)


samples_filename = st.file_uploader("Загрузите файлы спектров (.txt, .csv, .xls, .pts)",
                                    type=['txt', 'csv', 'xls', 'pts'], accept_multiple_files=True)
samples = {}
for i, sample_filename in enumerate(samples_filename):
    df = sers.data_read(sample_filename)
    df, _ = sers.baseline(df, lam=lam, itermax=itermax)
    samples[sample_filename.name] = df

peak_meas_samples = {}
for sample in samples.keys():
    peak_meas_samples[sample] = sers.get_inflection_points(
        samples[sample], peaks_points)

if len(samples) > 0:
    with st.expander(f"Вывод графиков"):
        selection = st.pills("Выберете файлы для вывода",
                             options=samples.keys(),
                             selection_mode="single")
        if selection is not None:
            fig = px.line(samples[selection])
            for peak in peaks_points:
                print(peak)
                fig.add_shape(
                    type="rect",
                    x0=peak[0],  # Начало области по X
                    x1=peak[1],  # Конец области по X
                    y0=0,    # Нижняя граница по Y (0 = минимум)
                    y1=1,
                    # Используем относительные координаты по Y (0-1)
                    yref="paper",
                    fillcolor="lightblue",
                    opacity=0.3,    # Прозрачность
                    line=dict(width=0),  # Убираем границу
                    layer="below"  # Размещаем под данными
                )
            st.plotly_chart(fig)
            st.dataframe(peak_meas_samples[selection])
            for i in range(len(peaks_points)):
                st.text(
                    f"Пик {i+1}: {peak_meas_samples[selection][f"Y{i+1}"].count()} / {peak_meas_samples[selection].shape[0]}")

    res_samples_dict = {}
    for peak in range(len(peaks_points)):
        res_samples = pd.DataFrame()
        for peak_meas_sample in peak_meas_samples.keys():
            res_samples[peak_meas_sample] = peak_meas_samples[
                peak_meas_sample][f"Y{peak+1}"]
        res_samples_dict[peak+1] = res_samples
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.title.set_text(f"Пик {peak+1}")
        ax.boxplot(res_samples.dropna())
        st.pyplot(fig)

        res_mean_std = pd.DataFrame(
            index=["mean", "median", "std"])
        for sample in res_samples_dict[peak+1].columns:
            res_mean_std[sample] = [res_samples_dict[peak+1][sample].mean(),
                                    res_samples_dict[peak+1][sample].median(),
                                    res_samples_dict[peak+1][sample].std()]

        st.dataframe(res_mean_std)

    with st.expander(f"Вывод результаов по пику"):
        selection = st.pills("Выберете пик",
                             options=res_samples_dict.keys(),
                             selection_mode="single")
        if selection is not None:
            st.dataframe(res_samples_dict[selection])
