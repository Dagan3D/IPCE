import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq, irfft
from io import StringIO
import os

# Функции
def read_file(uploaded_file) -> pd.DataFrame:
    """Читает файл с данными спектра и возвращает DataFrame."""
    try:
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
            raise ValueError("Неподдерживаемый формат файла.")
        return df
    except Exception as e:
        st.error(f"Ошибка при чтении файла {uploaded_file.name}: {e}")
        return None

def data_correction(df, correction_list, smooth):
    """Исправление скачка на спектрофотометре."""
    nm = df["Длина волны, нм"]
    df = df.copy()
    for corr in correction_list:
        for col in df.columns[1:]:
            if (corr + 1) in nm.values and corr in nm.values:
                diff = df.loc[nm == corr+1, col].values[0] - df.loc[nm == corr, col].values[0]
                df.loc[nm > corr, col] = df.loc[nm > corr, col] - diff

    if smooth > 0:
        for col in df.columns[1:]:
            df[col] = df[col].ewm(span=smooth).mean()
    return df


def cut_data(df, start, end):
    """Обрезка данных."""
    return df[(df["Длина волны, нм"] >= start) & (df["Длина волны, нм"] <= end)]


def zero_correction(df):
    """Выравнивание первой точки в 0."""
    df = df.copy()
    for col in df.columns[1:]:
        df.loc[:, col] = df[col] - df[col].iloc[0]
    return df


def smooth(sample, fft_cutoff_low, fft_cutoff_high):
    """Сглаживание с помощью преобразования Фурье."""
    yf = rfft(sample.values)
    yf[fft_cutoff_low:fft_cutoff_high] = 0
    return irfft(yf)



def respons_line(col, smooth_param, baseline_start, baseline_end, response_start, response_end, FFT_min, FFT_max):
    """Вычисление базовой линии и отклика."""
    dff = pd.DataFrame()
    dff["x"] = smooth(col.ewm(span=smooth_param).mean(), FFT_min, FFT_max)

    original_index = col.index
    start_idx = (pd.Series(original_index) - data_range[0]).abs().argmin()
    end_idx = (pd.Series(original_index) - data_range[1]).abs().argmin()
    trimmed_index = original_index[start_idx:end_idx+1]

    if len(trimmed_index) != len(dff["x"]):
       new_index = np.linspace(trimmed_index.min(), trimmed_index.max(), len(dff["x"]))
    else:
        new_index = trimmed_index

    dff.index = new_index

    if baseline_start < dff.index.min():
        baseline_start = dff.index.min()
        st.warning(f"baseline_start скорректирован до: {baseline_start}")
    elif baseline_start > dff.index.max():
        baseline_start = dff.index.max()
        st.warning(f"baseline_start скорректирован до: {baseline_start}")

    if baseline_end < dff.index.min():
        baseline_end = dff.index.min()
        st.warning(f"baseline_end скорректирован до: {baseline_end}")
    elif baseline_end > dff.index.max():
        baseline_end = dff.index.max()
        st.warning(f"baseline_end скорректирован до: {baseline_end}")

    if response_start < dff.index.min():
        response_start = dff.index.min()
        st.warning(f"response_start скорректирован до: {response_start}")
    elif response_start > dff.index.max():
        response_start = dff.index.max()
        st.warning(f"response_start скорректирован до: {response_start}")

    if response_end < dff.index.min():
        response_end = dff.index.min()
        st.warning(f"response_end скорректирован до: {response_end}")
    elif response_end > dff.index.max():
        response_end = dff.index.max()
        st.warning(f"response_end скорректирован до: {response_end}")

    x1_idx = (pd.Series(dff.index) - baseline_start).abs().argmin()
    x2_idx = (pd.Series(dff.index) - baseline_end).abs().argmin()
    x1, y1 = dff.index[x1_idx], dff["x"].iloc[x1_idx]
    x2, y2 = dff.index[x2_idx], dff["x"].iloc[x2_idx]

    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    dff["x" + " line"] = a * dff.index + b
    dff["x" + " base"] = dff["x" + " line"] - dff["x"]

    response_start_idx = (pd.Series(dff.index) - response_start).abs().argmin()
    response_end_idx = (pd.Series(dff.index) - response_end).abs().argmin()

    return dff.iloc[response_start_idx:response_end_idx]

def response_calc(dff):
    """Вычисляет суммарный отклик."""
    return dff["x base"].sum()

def make_non_negative(df: pd.DataFrame):
    """Сдвигает все значения DataFrame, делая их неотрицательными."""
    min_val = df.min().min()  # Находим минимальное значение во всем DataFrame
    if min_val < 0:
        return df - min_val
    return df


# --- Интерфейс Streamlit ---

st.title("Анализ поглощения")

uploaded_files = st.file_uploader("Загрузите файлы спектров (.txt, .csv, .xls, .pts)", type=['txt', 'csv', 'xls', 'pts'], accept_multiple_files=True)

if not uploaded_files:
    st.stop()

data = {}
samples = []
concentrations_dict = {}

for uploaded_file in uploaded_files:
    df = read_file(uploaded_file)
    if df is not None:
        file_name = os.path.splitext(uploaded_file.name)[0]
        try:
            sample_name, concentration_str = file_name.split(" ")
            if concentration_str == "0":
                concentration = 0.0
            else:
                concentration = 10 ** (-int(concentration_str.split("-")[1]))
        except (ValueError, IndexError):
            st.warning(f"Некорректный формат имени файла: {uploaded_file.name}.")
            continue
        if sample_name not in samples: samples.append(sample_name)
        if sample_name not in concentrations_dict: concentrations_dict[sample_name] = []
        concentrations_dict[sample_name].append(concentration)
        df.rename(columns={"Интенсивность": file_name}, inplace=True)
        data[file_name] = df

for sample_name in concentrations_dict:
    concentrations_dict[sample_name] = sorted(concentrations_dict[sample_name])

st.sidebar.header("Параметры обработки")
correct_jumps = st.sidebar.checkbox("Убрать скачки", value=True)
correction_points = st.sidebar.text_input("Точки коррекции", "339, 340, 387, 388, 389, 390, 453, 565")
correction_list = [int(x.strip()) for x in correction_points.split(',') if x.strip()]
smooth_data = st.sidebar.checkbox("Сглаживание данных", value=True)
smooth_param = st.sidebar.slider("Параметр сглаживания", 1, 50, 4) if smooth_data else 0
st.sidebar.header("Обрезка данных")
data_range = st.sidebar.slider("Диапазон длин волн (нм)", 100, 1500, (190, 1100))
zero_corr = st.sidebar.checkbox("Нулевая коррекция", value=True)
st.sidebar.header("Параметры расчета отклика")
fft_cutoff_low = st.sidebar.slider("Нижняя граница FFT", 0, 10, 4, key="fft_low")
fft_cutoff_high = st.sidebar.slider("Верхняя граница FFT", 30, 300, 150, key="fft_high")
baseline_start = st.sidebar.number_input("Начало базовой линии (нм)", value=485)
baseline_end = st.sidebar.number_input("Конец базовой линии (нм)", value=645)
response_start = st.sidebar.number_input("Начало области отклика (нм)", value=500)
response_end = st.sidebar.number_input("Конец области отклика (нм)", value=580)

for file_name, df in data.items():
     if correct_jumps:
         data[file_name] = data_correction(df, correction_list, smooth_param if smooth_data else 0)
     data[file_name] = cut_data(data[file_name], data_range[0], data_range[1])
     if zero_corr:
         data[file_name] = zero_correction(data[file_name])

combined_df = pd.DataFrame()
combined_df["Длина волны, нм"] = data[list(data.keys())[0]]["Длина волны, нм"]
for file_name, df in data.items():
    combined_df = combined_df.merge(data[file_name], on="Длина волны, нм", how="left")
combined_df = combined_df.set_index("Длина волны, нм")

"""### Визуализация данных"""
st.subheader("Предобработанные данные")
fig_raw = go.Figure()
for file_name in data.keys():
    fig_raw.add_trace(go.Scatter(x=combined_df.index, y=combined_df[file_name], mode='lines', name=file_name))
fig_raw.update_layout(xaxis_title="Длина волны, нм", yaxis_title="Отражение/Пропускание, %", legend_title="Образцы")
st.plotly_chart(fig_raw)

"""### Результаты расчета отклика"""
res = pd.DataFrame(columns=["Концентрация, М"] + samples)
res = res.set_index("Концентрация, М")

for sample in samples:
    sample_responses = {}
    with st.expander(f"Графики и отклик для {sample}", expanded=True):
        for concentration in concentrations_dict[sample]:
            if concentration == 0:  file_name = f"{sample} 0"
            else:
                conc_str = f"{concentration:.0e}".replace("e-0", "e-")
                file_name = f"{sample} {conc_str.split('e')[0].replace('.0','')}{conc_str.split('e')[1]}"

            if file_name in combined_df.columns:
                col = combined_df[file_name]
                dff = respons_line(col, smooth_param, baseline_start, baseline_end, response_start, response_end, fft_cutoff_low, fft_cutoff_high)
                sample_responses[concentration] = response_calc(dff)
                fig_sample = go.Figure()
                fig_sample.add_trace(go.Scatter(x=col.index, y=col, mode='lines', name=f'Исходный ({file_name})'))
                fig_sample.add_trace(go.Scatter(x=dff.index, y=dff['x'], mode='lines', name=f'Сглаженный ({file_name})'))
                fig_sample.add_trace(go.Scatter(x=dff.index, y=dff['x line'], mode='lines', name=f'Базовая линия ({file_name})'))
                title_conc = concentration if concentration != 0 else 0
                fig_sample.update_layout(
                    xaxis_title="Длина волны, нм", yaxis_title="Отражение/Пропускание",
                    title=f"{sample} - Концентрация: {title_conc}", legend=dict(x=0, y=1), hovermode="x unified")
                st.plotly_chart(fig_sample)
            else:
                best_match = None
                for col_name in combined_df.columns:
                    if col_name.startswith(sample):
                        try:
                            _, col_conc_str = col_name.split(" ")
                            col_conc = 0.0 if col_conc_str == "0" else 10**(-int(col_conc_str.split("-")[1]))
                            if col_conc == concentration:
                                best_match = col_name
                                break
                        except: pass
                if best_match:
                    col = combined_df[best_match]
                    dff = respons_line(col, smooth_param, baseline_start, baseline_end, response_start, response_end, fft_cutoff_low, fft_cutoff_high)
                    sample_responses[concentration] = response_calc(dff)
                    fig_sample = go.Figure()
                    fig_sample.add_trace(go.Scatter(x=col.index, y=col, mode='lines', name=f'Исходный ({best_match})'))
                    fig_sample.add_trace(go.Scatter(x=dff.index, y=dff['x'], mode='lines', name=f'Сглаженный ({best_match})'))
                    fig_sample.add_trace(go.Scatter(x=dff.index, y=dff['x line'], mode='lines', name=f'Базовая линия ({best_match})'))
                    title_conc = concentration if concentration != 0 else 0
                    fig_sample.update_layout(
                        xaxis_title="Длина волны, нм", yaxis_title="Отражение/Пропускание",
                        title=f"{sample} - Концентрация: {title_conc}", legend=dict(x=0, y=1), hovermode="x unified")
                    st.plotly_chart(fig_sample)
                else:
                    title_conc = concentration if concentration != 0 else 0
                    st.warning(f"Файл для {sample} и конц. {title_conc} М не найден.")
        st.write(f"Суммарный отклик для образца {sample}:")
        res[sample] = pd.Series(sample_responses)

# Опция "Сделать неотрицательными"


st.write("Суммарный отклик (все образцы):")
st.dataframe(res)

make_positive = st.checkbox("Сделать значения неотрицательными", value=False)
if make_positive:
    res = make_non_negative(res)
log_x = st.checkbox("Логарифмический масштаб по оси X", value=True)
log_y = st.checkbox("Логарифмический масштаб по оси Y", value=False)

fig_response = go.Figure()
for sample in samples:
    fig_response.add_trace(go.Scatter(x=res.index, y=res[sample], mode='lines+markers', name=sample))

fig_response.update_layout(
    xaxis_title="Концентрация аналита, М", yaxis_title="Суммарное поглощение", legend_title="Образцы",
    xaxis_type="log" if log_x else "linear", yaxis_type="log" if log_y else "linear")
st.plotly_chart(fig_response)

@st.cache_data
def convert_df(df):
    return df.to_csv(sep=";", index=True).encode('cp1251')

csv = convert_df(res)
st.download_button(label="Скачать результаты", data=csv, file_name='response.csv', mime='text/csv', type="primary")