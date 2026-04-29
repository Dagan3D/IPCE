import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import rfft, irfft
import os

# --- Функции чтения и обработки ---
def read_file(uploaded_file) -> pd.DataFrame:
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".txt":
            dataframe = pd.read_table(uploaded_file)
        elif ext == ".csv":
            dataframe = pd.read_csv(uploaded_file, sep=";")
        elif ext == ".xls":
            dataframe = pd.ExcelFile(uploaded_file).parse(0)
        elif ext == ".pts":
            dataframe = pd.read_table(uploaded_file, encoding="cp1251", sep="  ", 
                                      engine="python", skiprows=14, decimal='.').dropna().reset_index()
        elif ext == ".sf":
            dataframe = pd.read_table(uploaded_file, encoding="cp1251", header=None, 
                                      sep=r"\s+", engine="python", skiprows=18, skipfooter=4).dropna().reset_index()
        else:
            return None

        df = pd.DataFrame()
        if ext in[".pts", ".sf"]:
            df["Длина волны, нм"] = dataframe.iloc[:, 1].astype(float)
            df["Интенсивность"] = dataframe.iloc[:, 2].astype(float)
        else:
            df["Длина волны, нм"] = dataframe.iloc[:, 0].astype(float)
            df["Интенсивность"] = dataframe.iloc[:, 1].astype(float)
        return df
    except Exception as e:
        st.error(f"Ошибка при чтении {uploaded_file.name}: {e}")
        return None

def data_correction(df, correction_list):
    nm = df["Длина волны, нм"]
    df = df.copy()
    for corr in correction_list:
        for col in df.columns[1:]:
            if (corr + 1) in nm.values and corr in nm.values:
                val1 = df.loc[nm == corr+1, col].values[0]
                val2 = df.loc[nm == corr, col].values[0]
                diff = val1 - val2
                df.loc[nm > corr, col] = df.loc[nm > corr, col] - diff
    return df

def cut_data(df, start, end):
    return df[(df["Длина волны, нм"] >= start) & (df["Длина волны, нм"] <= end)]

def smooth_fft(sample, fft_cutoff_low, fft_cutoff_high):
    yf = rfft(sample.values)
    yf[int(fft_cutoff_low):int(fft_cutoff_high)] = 0
    return irfft(yf)

def respons_line(col, smooth_param, baseline_start, baseline_end, response_start, response_end, FFT_min, FFT_max, data_range, use_fft):
    dff = pd.DataFrame()
    smoothed_val = col.ewm(span=smooth_param).mean() if smooth_param > 0 else col.copy()
    
    if use_fft:
        dff["x"] = smooth_fft(smoothed_val, FFT_min, FFT_max)
        new_index = np.linspace(col.index.min(), col.index.max(), len(dff["x"]))
        dff.index = new_index
    else:
        dff["x"] = smoothed_val.values
        dff.index = col.index

    x1_idx = (pd.Series(dff.index) - baseline_start).abs().argmin()
    x2_idx = (pd.Series(dff.index) - baseline_end).abs().argmin()
    x1, y1 = dff.index[x1_idx], dff["x"].iloc[x1_idx]
    x2, y2 = dff.index[x2_idx], dff["x"].iloc[x2_idx]

    a = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
    b = y1 - a * x1
    dff["x line"] = a * dff.index + b
    dff["x base"] = dff["x line"] - dff["x"]

    res_start_idx = (pd.Series(dff.index) - response_start).abs().argmin()
    res_end_idx = (pd.Series(dff.index) - response_end).abs().argmin()
    return dff.iloc[res_start_idx:res_end_idx]

def response_calc(dff):
    return dff["x base"].sum()

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(sep=";", index=True).encode('utf-8-sig')

# --- Интерфейс Streamlit ---

st.set_page_config(page_title="Анализ поглощения", layout="wide")
st.title("🧪 Анализ спектров поглощения")

uploaded_files = st.file_uploader("1. Загрузите файлы спектров (включая эталон)", 
                                  type=['txt', 'csv', 'xls', 'pts', 'sf'], 
                                  accept_multiple_files=True)

if uploaded_files:
    st.subheader("2. Настройка эталона")
    use_etalon = st.checkbox("Делить спектры на эталон?", value=True)
    etalon_filename = None
    if use_etalon:
        file_names =[f.name for f in uploaded_files]
        etalon_filename = st.selectbox("Выберите файл эталона (Blank / Reference):", file_names)

    st.subheader("3. Настройка параметров образцов")
    st.info("💡 Оставьте столбцы **Старт базы** и **Конец базы** пустыми, чтобы использовать глобальные значения из левого меню.")
    file_registry =[]
    
    for f in uploaded_files:
        if use_etalon and f.name == etalon_filename:
            continue
            
        fname = os.path.splitext(f.name)[0]
        suggested_sample = fname.split(" ")[0]
        suggested_conc = 0.0
        try:
            if " " in fname:
                parts = fname.split(" ")
                if parts[1] == "0":
                    suggested_conc = 0.0
                elif "-" in parts[1]:
                    suggested_conc = 10 ** (-float(parts[1].split("-")[1]))
        except:
            pass
        
        file_registry.append({
            "Имя файла": f.name,
            "Образец": suggested_sample,
            "Концентрация (M)": suggested_conc,
            "Старт базы (нм)": None, 
            "Конец базы (нм)": None  
        })

    if not file_registry:
        st.warning("Нет образцов для анализа (выбран только эталон).")
        st.stop()

    input_df = pd.DataFrame(file_registry)
    edited_config = st.data_editor(
        input_df, 
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Концентрация (M)": st.column_config.NumberColumn(format="%.2e"),
            "Старт базы (нм)": st.column_config.NumberColumn(help="Оставьте пустым"),
            "Конец базы (нм)": st.column_config.NumberColumn(help="Оставьте пустым")
        }
    )

    data = {}
    samples = edited_config["Образец"].unique().tolist()
    
    for i, row in edited_config.iterrows():
        file_obj = next(f for f in uploaded_files if f.name == row["Имя файла"])
        df = read_file(file_obj)
        if df is not None:
            col_name = f"{row['Образец']} {row['Концентрация (M)']:.1e}"
            df = df.rename(columns={"Интенсивность": col_name})
            
            data[col_name] = {
                "df": df, 
                "sample": row["Образец"], 
                "conc": row["Концентрация (M)"],
                "b_start": row["Старт базы (нм)"],
                "b_end": row["Конец базы (нм)"]
            }

    if use_etalon and etalon_filename:
        file_obj = next(f for f in uploaded_files if f.name == etalon_filename)
        etalon_df = read_file(file_obj)
        if etalon_df is not None:
            etalon_df = etalon_df.rename(columns={"Интенсивность": "__ETALON__"})
            data["__ETALON__"] = {"df": etalon_df}

    # --- Сайдбар с настройками ---
    st.sidebar.header("Параметры обработки")
    correct_jumps = st.sidebar.checkbox("Убрать скачки", value=False) # Выключил по умолчанию!
    correction_points = st.sidebar.text_input("Точки коррекции", "339, 340, 387, 388, 389, 390, 453, 565, 700, 701, 702")
    correction_list =[int(x.strip()) for x in correction_points.split(',') if x.strip()]
    
    smooth_data = st.sidebar.checkbox("Сглаживание данных (EWM)", value=True)
    smooth_p = st.sidebar.slider("Параметр сглаживания (EWM)", 1, 50, 4) if smooth_data else 0
    
    data_range = st.sidebar.slider("Диапазон длин волн (нм)", 100, 1500, (478, 744))
    
    # НОВОЕ: Выпадающий список для умной нулевой коррекции
    zero_corr_mode = st.sidebar.selectbox(
        "Нулевая коррекция (сдвиг по Y)",["Не применять", "По первой точке (начало в 0)", "По минимуму (убрать минусы)"],
        index=2
    )
    
    st.sidebar.header("Параметры FFT и Базовой линии")
    use_fft = st.sidebar.checkbox("Применить FFT сглаживание", value=True)
    
    if use_fft:
        fft_low = st.sidebar.slider("Нижняя граница FFT", 0, 10, 4)
        fft_high = st.sidebar.slider("Верхняя граница FFT", 30, 300, 150)
    else:
        fft_low, fft_high = 0, 0
        
    b_start_global = st.sidebar.number_input("Глобальное начало базы (нм)", value=485)
    b_end_global = st.sidebar.number_input("Глобальный конец базы (нм)", value=645)

    if data:
        processed_data = {}
        for key, item in data.items():
            df = item["df"].copy()
            if correct_jumps:
                df = data_correction(df, correction_list)
            df = cut_data(df, data_range[0], data_range[1])
            processed_data[key] = df

        combined_df = pd.DataFrame()
        first_key = list(processed_data.keys())[0]
        combined_df["Длина волны, нм"] = processed_data[first_key]["Длина волны, нм"]
        
        for key, df in processed_data.items():
            combined_df = combined_df.merge(df, on="Длина волны, нм", how="outer")
        
        combined_df = combined_df.set_index("Длина волны, нм").sort_index()
        combined_df = combined_df.interpolate(method='index').dropna(how='all')

        if use_etalon and "__ETALON__" in combined_df.columns:
            etalon_col = combined_df["__ETALON__"].replace(0, np.nan)
            for col in combined_df.columns:
                if col != "__ETALON__":
                    combined_df[col] = combined_df[col] / etalon_col
            combined_df = combined_df.drop(columns=["__ETALON__"])

        # НОВАЯ ЛОГИКА НУЛЕВОЙ КОРРЕКЦИИ
        if zero_corr_mode != "Не применять":
            for col in combined_df.columns:
                if zero_corr_mode == "По первой точке (начало в 0)":
                    first_valid_idx = combined_df[col].first_valid_index()
                    if first_valid_idx is not None:
                        combined_df[col] = combined_df[col] - combined_df[col].loc[first_valid_idx]
                elif zero_corr_mode == "По минимуму (убрать минусы)":
                    min_val = combined_df[col].min()
                    combined_df[col] = combined_df[col] - min_val

        st.subheader("4. Предобработанные данные (без сглаживания)")
        fig_raw = go.Figure()
        for col in combined_df.columns:
            fig_raw.add_trace(go.Scatter(x=combined_df.index, y=combined_df[col], mode='lines', name=col))
        
        y_axis_title = "Относительное пропускание/отражение" if use_etalon else "Интенсивность"
        fig_raw.update_layout(xaxis_title="Длина волны, нм", yaxis_title=y_axis_title)
        st.plotly_chart(fig_raw, use_container_width=True)

        st.download_button(
            label="📥 Скачать предобработанные спектры (CSV)", 
            data=convert_df_to_csv(combined_df),
            file_name='processed_spectra.csv', 
            mime='text/csv'
        )

        st.subheader("5. Результаты расчета")
        res_list =[]

        for sample in samples:
            with st.expander(f"Графики для {sample}", expanded=False):
                sample_cols =[c for c in combined_df.columns if c.startswith(f"{sample} ")]
                
                for col_name in sample_cols:
                    col_data = combined_df[col_name].dropna()
                    if col_data.empty:
                        continue
                    
                    ind_start = data[col_name]["b_start"]
                    ind_end = data[col_name]["b_end"]
                    
                    current_b_start = b_start_global if pd.isna(ind_start) else float(ind_start)
                    current_b_end = b_end_global if pd.isna(ind_end) else float(ind_end)
                    
                    dff = respons_line(
                        col_data, smooth_p, 
                        current_b_start, current_b_end, 
                        current_b_start, current_b_end, 
                        fft_low, fft_high, data_range, use_fft
                    )
                    val = response_calc(dff)
                    
                    conc = data[col_name]["conc"]
                    res_list.append({"Образец": sample, "Концентрация, М": conc, "Отклик": val})
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=col_data.index, y=col_data, name="Исходный (нормированный)"))
                    
                    smooth_name = "Сглаженный (FFT)" if use_fft else "Сглаженный (без FFT)"
                    fig.add_trace(go.Scatter(x=dff.index, y=dff["x"], name=smooth_name))
                    fig.add_trace(go.Scatter(x=dff.index, y=dff["x line"], name="База"))
                    
                    fig.update_layout(
                        title=f"{col_name} (Отклик: {val:.2f}) | База: {current_b_start} - {current_b_end} нм", 
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

        if res_list:
            res_df = pd.DataFrame(res_list)
            pivot_res = res_df.pivot(index="Концентрация, М", columns="Образец", values="Отклик")
            
            st.write("Сводная таблица отклика:")
            st.dataframe(pivot_res)

            st.download_button(
                label="📥 Скачать таблицу откликов (CSV)", 
                data=convert_df_to_csv(pivot_res),
                file_name='response_results.csv', 
                mime='text/csv',
                type="primary"
            )

            st.subheader("6. Калибровочный график")
            log_x = st.checkbox("Логарифмический X", value=True)
            make_positive = st.checkbox("Сделать значения на графике неотрицательными", value=False)
            
            fig_res = go.Figure()
            for s in samples:
                s_data = res_df[res_df["Образец"] == s].sort_values("Концентрация, М").copy()
                
                if make_positive:
                    min_val = s_data["Отклик"].min()
                    if min_val < 0:
                        s_data["Отклик"] = s_data["Отклик"] - min_val

                fig_res.add_trace(go.Scatter(x=s_data["Концентрация, М"], y=s_data["Отклик"], 
                                             mode='lines+markers', name=s))
            
            fig_res.update_layout(xaxis_type="log" if log_x else "linear", 
                                  xaxis_title="Концентрация, М", yaxis_title="Суммарный отклик")
            st.plotly_chart(fig_res, use_container_width=True)

else:
    st.info("Пожалуйста, загрузите файлы спектров для начала работы.")