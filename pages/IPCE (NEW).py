import streamlit as st
from scipy import interpolate
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os
import to_IPCE

# Импортируем метод ALS для базовой линии
from SERS_analis.als import als

@st.cache_data
def convert_df(df):
    return df.to_csv(sep=";", index=False).encode('cp1251')

def extract_photocurrents_from_signal(df, p_type="Анодный (пики вверх)", lam=1e6, p_val=0.01):
    """Функция для извлечения фототока с помощью ALS"""
    y = df["Current"].values.copy()
    
    if p_type == "Катодный (пики вниз)":
        y = -y
        
    baseline = als(y, lam=lam, p=p_val, itermax=10)
    photocurrent_signal = y - baseline
    
    if p_type == "Катодный (пики вниз)":
        baseline = -baseline
        photocurrent_signal = -photocurrent_signal
        
    df["Baseline"] = baseline
    df["Photocurrent_Signal"] = photocurrent_signal
    
    extracted_data = []
    waves = df["Wavelength"].unique()
    
    for w in waves:
        chunk = df[df["Wavelength"] == w]
        if p_type == "Анодный (пики вверх)":
            peak_val = np.percentile(chunk["Photocurrent_Signal"], 95)
        else:
            peak_val = np.abs(np.percentile(chunk["Photocurrent_Signal"], 5))
            
        extracted_data.append({"Длина волны, нм": w, "Photocurrent_A": peak_val})
        
    return df, pd.DataFrame(extracted_data)


st.title("IPCE (Обработка временных рядов с помощью ALS)")

"""## Обработка данных со спектрофотометра для получения зависимости IPCE от длины волны"""

# --- Калибровочный график ---
with st.expander("Калибровочный график", expanded=True):
    st.markdown('''
    **Физика расчета:** Калибровочный фотодиод имеет площадь 1 см², поэтому измеряемая мощность 
    численно равна **плотности мощности (Вт/см²)** падающего света.
    ''')
    
    calibration_valid = False
    calibration_get = st.selectbox(
        "У вас есть готовый файл калибровки?",
        ("Получить из файла данных (сырые данные эталона)", "Загрузить готовый файл калибровки"),
    )
    
    if (calibration_get == "Получить из файла данных (сырые данные эталона)"):
        uploaded_file = st.file_uploader("Файл данных эталона", type = ['txt', 'csv'])
        
        if uploaded_file is not None:
            df_calib_raw = to_IPCE.read_data(uploaded_file)
            df_calib_raw = to_IPCE.reduction_smooth(df_calib_raw, window=1)
            df_calib_raw = to_IPCE.time_split(df_calib_raw, start_wave=280)
            
            df_calib_raw, df_calib_peaks = extract_photocurrents_from_signal(df_calib_raw, lam=1e6, p_val=0.01)
            
            # РАСЧЕТ ПЛОТНОСТИ МОЩНОСТИ (Ток в Амперах / Чувствительность в А/Вт = Мощность в Ваттах)
            # Т.к. датчик 1 см2, Ватты = Вт/см2
            df_calib_peaks["K_diode_A_W"] = df_calib_peaks['Длина волны, нм'].apply(to_IPCE.p_diode)
            df_calib_peaks["Плотность_мощности_Вт_см2"] = df_calib_peaks["Photocurrent_A"] / df_calib_peaks["K_diode_A_W"]
            df_calib_peaks["Плотность_мощности_мкВт_см2"] = df_calib_peaks["Плотность_мощности_Вт_см2"] * 1e6 
            
            calibration_valid = True
            linear_calib = interpolate.interp1d(df_calib_peaks["Длина волны, нм"], df_calib_peaks["Плотность_мощности_Вт_см2"], kind="linear", fill_value="extrapolate")
            
            fig = px.line(df_calib_peaks, x="Длина волны, нм", y="Плотность_мощности_мкВт_см2", title="Плотность мощности излучения")
            fig.update_layout(yaxis_title="Мощность, мкВт/см²")
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            
            csv = convert_df(df_calib_peaks)
            st.download_button(label="Скачать файл калибровки", data=csv, file_name='Calibration.csv', mime='text/csv')

    else:
        uploaded_file = st.file_uploader("Готовый файл калибровки (.csv)", type = ['txt', 'csv'])
        if uploaded_file is not None:
            df_calib_peaks = pd.read_table(uploaded_file, sep=';', encoding="cp1251")
            calibration_valid = True
            
            if "Wavelength" in df_calib_peaks.columns:
                df_calib_peaks.rename(columns={"Wavelength": "Длина волны, нм"}, inplace=True)
                
            # Используем колонку с ВАТТАМИ для расчетов
            linear_calib = interpolate.interp1d(df_calib_peaks["Длина волны, нм"], df_calib_peaks["Плотность_мощности_Вт_см2"], kind="linear", fill_value="extrapolate")
            
            fig = px.line(df_calib_peaks, x="Длина волны, нм", y="Плотность_мощности_мкВт_см2", title="Плотность мощности излучения")
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


# --- Файлы данных ---
if calibration_valid:
    raw_data_dict = {}
    
    with st.expander("Загрузка файлов данных образцов"):
        uploaded_files = st.file_uploader("Файлы данных образцов", type=['txt', 'csv'], accept_multiple_files=True)
        
        for uploaded_file in uploaded_files:
            df_raw = to_IPCE.read_data(uploaded_file)
            df_raw = to_IPCE.time_split(df_raw, start_wave=280)
            df_raw = to_IPCE.reduction_smooth(df_raw)
            sample_name = os.path.splitext(uploaded_file.name)[0]
            raw_data_dict[sample_name] = df_raw
            
        if len(raw_data_dict) > 0:
            fig_raw = go.Figure()
            for name, df in raw_data_dict.items():
                fig_raw.add_trace(go.Scatter(x=df["Time"], y=df["Current"]*1e6, mode='lines', name=name))
            fig_raw.update_layout(title="Исходные токи", xaxis_title="Время, с", yaxis_title="Сила тока, мкА")
            st.plotly_chart(fig_raw, use_container_width=True)

    # --- Извлечение фототоков (ALS) ---
    if len(raw_data_dict) > 0:
        with st.expander("Настройка базовой линии (ALS) и извлечение фототоков", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                p_type = st.selectbox("Тип проводимости", ["Анодный (пики вверх)", "Катодный (пики вниз)"])
            with col2:
                lam_pow = st.slider("Жесткость линии (log10 λ)", 3.0, 10.0, 6.5, 0.5)
            with col3:
                p_val = st.number_input("Асимметрия (p)", 0.001, 0.5, 0.01, 0.005, format="%f")
            
            lam = 10 ** lam_pow
            
            processed_signals = {}
            currents_abs_A = pd.DataFrame() 
            
            for i, (name, df) in enumerate(raw_data_dict.items()):
                df_processed, df_peaks = extract_photocurrents_from_signal(df, p_type=p_type, lam=lam, p_val=p_val)
                processed_signals[name] = df_processed
                
                if i == 0:
                    currents_abs_A["Длина волны, нм"] = df_peaks["Длина волны, нм"]
                currents_abs_A[name] = df_peaks["Photocurrent_A"]

            sample_to_preview = st.selectbox("Выберите образец для проверки базовой линии:", list(processed_signals.keys()))
            df_preview = processed_signals[sample_to_preview]
            
            fig_als = go.Figure()
            fig_als.add_trace(go.Scatter(x=df_preview["Time"], y=df_preview["Current"]*1e6, name="Исходный сигнал", line=dict(color="lightgray")))
            fig_als.add_trace(go.Scatter(x=df_preview["Time"], y=df_preview["Baseline"]*1e6, name="Базовая линия (Фон)", line=dict(color="red", width=2)))
            fig_als.add_trace(go.Scatter(x=df_preview["Time"], y=df_preview["Photocurrent_Signal"]*1e6, name="Извлеченный Фототок", line=dict(color="blue")))
            fig_als.update_layout(title=f"Проверка ALS (в мкА): {sample_to_preview}", xaxis_title="Время, с", yaxis_title="Ток, мкА", hovermode="x unified")
            st.plotly_chart(fig_als, use_container_width=True)

        # --- Расчет IPCE ---
        with st.expander("Расчет IPCE", expanded=True):
            st.markdown(r'''$IPCE(\%) = \frac{1240 \cdot \mathbf{J} (А/см^2)}{\lambda (нм) \cdot \mathbf{P} (Вт/см^2)} \cdot 100$''')
            
            area_sample = st.number_input("Площадь освещаемого образца (см²)", min_value=0.01, value=1.0, step=0.1)
            
            ipce_df = pd.DataFrame()
            ipce_df["Длина волны, нм"] = currents_abs_A["Длина волны, нм"]
            
            density_df = pd.DataFrame()
            density_df["Длина волны, нм"] = currents_abs_A["Длина волны, нм"]
            
            # Плотность мощности света строго в Вт/см2
            power_density_W_cm2 = linear_calib(ipce_df["Длина волны, нм"])
            
            for name in processed_signals.keys():
                current_A = currents_abs_A[name]
                
                # Плотность тока в А/см2
                current_density_A_cm2 = current_A / area_sample
                
                # Расчет IPCE (Плотность тока / Плотность мощности)
                ipce_df[name] = (1240 * current_density_A_cm2) / (ipce_df["Длина волны, нм"] * power_density_W_cm2) * 100
                
                # Для красивого графика переводим в мкА/см2
                density_df[name] = current_density_A_cm2 * 1e6
            
            fig_ipce = px.line(ipce_df, x="Длина волны, нм", y=list(processed_signals.keys()), title="Квантовая эффективность (IPCE)")
            fig_ipce.update_layout(yaxis_title="IPCE, %", xaxis_title="Длина волны, нм")
            st.plotly_chart(fig_ipce, use_container_width=True)
            
            fig_dens = px.line(density_df, x="Длина волны, нм", y=list(processed_signals.keys()), title="Плотность фототока (мкА/см²)")
            fig_dens.update_layout(yaxis_title="Плотность тока, мкА/см²")
            st.plotly_chart(fig_dens, use_container_width=True)
            
            if st.checkbox('Показать таблицу IPCE'):
                st.dataframe(ipce_df)

            csv_ipce = convert_df(ipce_df)
            st.download_button(
                label="Скачать результат (IPCE.csv)",
                data=csv_ipce,
                file_name='IPCE_Results.csv',
                mime='text/csv',
                type="primary"
            )