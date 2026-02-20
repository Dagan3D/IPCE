import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.stats import linregress


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


def find_derivative_maxima(hv, y, min_distance=10):
    """
    Находит точки максимума производной графика Таука.
    
    Parameters:
    -----------
    hv : array-like
        Значения энергии (eV)
    y : array-like
        Значения (F*hv)^degree
    min_distance : int
        Минимальное расстояние между пиками в точках
    
    Returns:
    --------
    list of dict: Список словарей с информацией о каждом максимуме
    """
    # Вычисляем производную
    dy = np.gradient(y, hv)
    
    # Находим пики (максимумы) производной
    peaks, properties = find_peaks(dy, distance=min_distance, prominence=0.01)
    
    maxima = []
    for peak_idx in peaks:
        maxima.append({
            'index': peak_idx,
            'hv': hv[peak_idx],
            'derivative': dy[peak_idx]
        })
    
    return maxima, dy


def fit_linear_region(hv, y, center_idx, window_size=10):
    """
    Линейная аппроксимация в окрестности точки.
    
    Parameters:
    -----------
    hv : array-like
        Значения энергии (eV)
    y : array-like
        Значения (F*hv)^degree
    center_idx : int
        Индекс центральной точки
    window_size : int
        Размер окна в одну сторону (всего 2*window_size + 1 точек)
    
    Returns:
    --------
    dict: Параметры аппроксимации (slope, intercept, r_squared, Eg)
    """
    # Определяем границы окна
    start_idx = max(0, center_idx - window_size)
    end_idx = min(len(hv), center_idx + window_size + 1)
    
    # Извлекаем данные в окне
    x_fit = hv[start_idx:end_idx]
    y_fit = y[start_idx:end_idx]
    
    # Убираем NaN значения
    mask = ~(np.isnan(x_fit) | np.isnan(y_fit))
    x_fit = x_fit[mask]
    y_fit = y_fit[mask]
    
    if len(x_fit) < 3:
        return None
    
    # Линейная регрессия
    result = linregress(x_fit, y_fit)
    
    # R² = r_value²
    r_squared = result.rvalue ** 2
    
    # Eg - точка пересечения с осью X (y = 0)
    # 0 = slope * Eg + intercept
    # Eg = -intercept / slope
    if result.slope != 0:
        Eg = -result.intercept / result.slope
    else:
        Eg = np.nan
    
    return {
        'slope': result.slope,
        'intercept': result.intercept,
        'r_squared': r_squared,
        'Eg': Eg,
        'x_fit': x_fit,
        'y_fit': y_fit
    }


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
            smooth_force = st.slider("Сила сглаживания", 1, 20, 4)
        else:
            smooth_force = 0

        for uploaded_file in uploaded_files:
            dataframe = read_file(uploaded_file)
            if smooth:
                dataframe = data_correction(dataframe, smooth=smooth_force)
            if dataframe is not None:
                samples.append(os.path.splitext(uploaded_file.name)[0])
                currents_sample['Длина волны, нм'] = dataframe["Длина волны, нм"]
                currents_sample[samples[-1]] = dataframe["Интенсивность"]
                data_valid = True

        if (data_valid):
            fig = px.line(currents_sample.dropna(), x="Длина волны, нм",
                          y=samples, labels={'value': "Отражение, %"})
            fig.update_layout(legend=dict(yanchor="top", xanchor="right"))
            st.plotly_chart(fig, theme="streamlit", width='stretch')
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
            st.plotly_chart(fig_ref, theme="streamlit", width='stretch')

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
            smooth_force_int = st.slider("Сила сглаживания", 1, 20, 4, key="smooth_force_int")
        else:
            smooth_force_int = 0

        if df_reference is not None and uploaded_files:
            # Объединяем данные эталона с образцами
            wavelengths = df_reference["Длина волны, нм"].values
            
            for uploaded_file in uploaded_files:
                df_sample = read_file(uploaded_file)
                
                if smooth_int:
                    df_sample = data_correction(df_sample, smooth=smooth_force_int)
                
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
            st.plotly_chart(fig_int, theme="streamlit", width='stretch')
            
            # Визуализация рассчитанного R
            st.subheader("Рассчитанный коэффициент (R = I/I₀)")
            fig_R = px.line(currents_sample.dropna(), x="Длина волны, нм",
                           y=samples, labels={'value': "R (отн. ед.)"})
            fig_R.update_layout(legend=dict(yanchor="top", xanchor="right"))
            st.plotly_chart(fig_R, theme="streamlit", width='stretch')
            
            if st.checkbox('Показать таблицу рассчитанных коэффициентов'):
                currents_sample

# %% Учёт площади образца
if (data_valid):
    with st.expander("Учёт площади образца"):
        """## Учёт площади образца и построение графика Таука
        Здесь производится расчёт оптической ширины запрещённой зоны методом Таука.
        """
        
        # Определяем диапазон данных для обрезки
        hv_values = 1240 / (currents_sample["Длина волны, нм"])
        hv_min = float(np.nanmin(hv_values))
        hv_max = float(np.nanmax(hv_values))
        
        # Слайдер для обрезки данных по энергии
        st.write("**Обрезка данных по энергии (hv):**")
        hv_range = st.slider(
            "Выберите диапазон энергии (eV)",
            min_value=round(hv_min, 2),
            max_value=round(hv_max, 2),
            value=(round(hv_min, 2), round(hv_max, 2)),
            step=0.01,
            key="hv_range_slider",
            help="Обрезает данные для расчёта. Полезно для удаления шумных участков спектра."
        )
        
        # Создаём маску для фильтрации данных
        mask = (hv_values >= hv_range[0]) & (hv_values <= hv_range[1])
        
        # Фильтруем данные
        filtered_wavelengths = currents_sample['Длина волны, нм'][mask].reset_index(drop=True)
        
        # Применяем фильтрацию к currents_sample для расчётов
        filtered_currents_sample = pd.DataFrame()
        filtered_currents_sample['Длина волны, нм'] = filtered_wavelengths
        for sample in samples:
            filtered_currents_sample[sample] = currents_sample[sample][mask].reset_index(drop=True)

        OpticEg = pd.DataFrame()
        OpticEg["Длина волны, нм"] = filtered_currents_sample['Длина волны, нм']
        OpticEg["hv"] = (1240 / (filtered_currents_sample["Длина волны, нм"]))

        degree = st.selectbox(
            'Какую степень использовать при вычислениях', (2, 1/2))
        x_asxi = st.selectbox("Подписи горизонтальной оси",
                              ('hv', 'Длина волны, нм'))

        for current_sample in samples:
            df_cub_munk = pd.DataFrame()
            df_cub_munk['hv'] = OpticEg["hv"]
            df_cub_munk['R'] = filtered_currents_sample[current_sample].values
            
            # Для режима интенсивностей R уже в относительных единицах
            # Для режима спектров R в процентах
            if load_mode == "Спектры (%)":
                df_cub_munk['R_frac'] = df_cub_munk['R'] / 100
            else:
                # R уже в относительных единицах (0-1), не нужно делить на 100
                df_cub_munk['R_frac'] = df_cub_munk['R']
            
            df_cub_munk['(1-R)^2'] = (1 - df_cub_munk['R_frac']) ** 2
            df_cub_munk['2R'] = df_cub_munk['R_frac'] * 2
            # Защита от деления на ноль
            df_cub_munk['k/s'] = np.where(
                df_cub_munk['2R'] != 0,
                df_cub_munk['(1-R)^2'] / df_cub_munk['2R'],
                np.nan
            )
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

        st.plotly_chart(fig, theme="streamlit", width='stretch')

        if st.checkbox('Показать таблицу результатов'):
            OpticEg

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(sep=";", index=False).encode('utf-8')

        csv = convert_df(OpticEg)

        st.download_button(
            label="Скачать результат",
            data=csv,
            file_name='OpticEg.csv',
            mime='text/csv',
            type="primary"
        )

    # %% Автоматическое определение Eg
    with st.expander("Автоматическое определение Eg"):
        """## Автоматическое определение оптической ширины запрещённой зоны
        
        Алгоритм находит максимумы производной графика Таука и строит 
        линейную аппроксимацию в окрестности каждого максимума.
        
        Каждый образец имеет индивидуальные настройки: выбор точки максимума и смещение.
        """
        
        # Инициализация session_state для хранения настроек каждого образца
        if 'eg_settings' not in st.session_state:
            st.session_state.eg_settings = {}
        
        # Инициализация настроек для всех образцов
        for sample in samples:
            if sample not in st.session_state.eg_settings:
                st.session_state.eg_settings[sample] = {
                    'maxima_idx': 0,
                    'offset': 0.0
                }
        
        # Выбор образца для анализа
        selected_sample = st.selectbox(
            "Выберите образец для анализа",
            samples,
            key="eg_sample_selector"
        )
        
        if selected_sample:
            # Получаем данные для выбранного образца
            hv = OpticEg["hv"].values
            y = OpticEg[selected_sample].values
            
            # Находим максимумы производной
            maxima, derivative = find_derivative_maxima(hv, y)
            
            if len(maxima) > 0:
                # Рассчитываем Eg и R² для каждого максимума
                for m in maxima:
                    fit_result = fit_linear_region(hv, y, m['index'], window_size=10)
                    if fit_result:
                        m['fit'] = fit_result
                        m['Eg'] = fit_result['Eg']
                        m['r_squared'] = fit_result['r_squared']
                    else:
                        m['fit'] = None
                        m['Eg'] = np.nan
                        m['r_squared'] = np.nan
                
                # Фильтруем максимумы с валидными результатами
                valid_maxima = [m for m in maxima if m.get('fit') is not None]
                
                if valid_maxima:
                    # Сортируем по R² (по убыванию)
                    valid_maxima.sort(key=lambda x: x['r_squared'], reverse=True)
                    
                    # Выбор максимума - используем сохранённое значение для данного образца
                    maxima_options = [
                        f"№{i+1}: Eg={m['Eg']:.3f} eV, R²={m['r_squared']:.4f}"
                        for i, m in enumerate(valid_maxima)
                    ]
                    
                    # Получаем сохранённый индекс максимума для данного образца
                    saved_maxima_idx = st.session_state.eg_settings[selected_sample]['maxima_idx']
                    # Убеждаемся, что индекс в допустимых пределах
                    saved_maxima_idx = min(saved_maxima_idx, len(maxima_options) - 1)
                    
                    selected_max_idx = st.selectbox(
                        "Выберите точку максимума",
                        range(len(maxima_options)),
                        format_func=lambda x: maxima_options[x],
                        index=saved_maxima_idx,
                        key=f"maxima_selector_{selected_sample}"
                    )
                    
                    # Сохраняем выбранный индекс максимума для данного образца
                    st.session_state.eg_settings[selected_sample]['maxima_idx'] = selected_max_idx
                    
                    selected_max = valid_maxima[selected_max_idx]
                    
                    # Слайдер для смещения в eV - используем сохранённое значение для данного образца
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        saved_offset = st.session_state.eg_settings[selected_sample]['offset']
                        offset = st.slider(
                            "Смещение точки (eV)",
                            min_value=-0.5,
                            max_value=0.5,
                            value=saved_offset,
                            step=0.01,
                            key=f"eg_offset_{selected_sample}"
                        )
                        # Сохраняем смещение для данного образца
                        st.session_state.eg_settings[selected_sample]['offset'] = offset
                    with col2:
                        # st.metric("Текущий Eg", f"{selected_max['Eg'] + offset:.3f} eV")
                    
                    # Рассчитываем аппроксимацию для смещённой точки
                        offset_hv = selected_max['hv'] + offset
                        
                        # Находим ближайший индекс к смещённой точке
                        offset_idx = np.argmin(np.abs(hv - offset_hv))
                        
                        # Пересчитываем аппроксимацию для смещённой точки
                        offset_fit = fit_linear_region(hv, y, offset_idx, window_size=10)
                        
                        if offset_fit:
                            st.markdown(
                                f"**Скорректированный Eg:** {offset_fit['Eg']:.3f} eV<br>"
                                f"**R²:** {offset_fit['r_squared']:.4f}",
                                 unsafe_allow_html=True)
                    
                    # Визуализация
                    fig_eg = go.Figure()
                    
                    # Основной график Таука
                    fig_eg.add_trace(go.Scatter(
                        x=hv, y=y,
                        mode='lines',
                        name=selected_sample,
                        line=dict(color='blue')
                    ))
                    
                    # Вертикальная линия выбранного Eg
                    if offset_fit and not np.isnan(offset_fit['Eg']):
                        fig_eg.add_vline(
                            x=offset_fit['Eg'],
                            line=dict(color='red', dash='dash'),
                            annotation=dict(text=f"Eg = {offset_fit['Eg']:.3f} eV")
                        )
                    
                    # Линейная аппроксимация
                    if offset_fit:
                        # Продлеваем линию влево до пересечения с осью X (Eg) и вправо на 1 эВ
                        x_line_left = offset_fit['Eg'] if not np.isnan(offset_fit['Eg']) else offset_fit['x_fit'].min()
                        x_line_right = offset_fit['x_fit'].max() + 0.05
                        x_line = np.linspace(x_line_left, x_line_right, 100)
                        y_line = offset_fit['slope'] * x_line + offset_fit['intercept']
                        fig_eg.add_trace(go.Scatter(
                            x=x_line, y=y_line,
                            mode='lines',
                            name=f'Аппроксимация (R²={offset_fit["r_squared"]:.3f})',
                            line=dict(color='green', dash='dot')
                        ))
                    
                    # Точка максимума производной
                    fig_eg.add_trace(go.Scatter(
                        x=[offset_hv],
                        y=[y[offset_idx] if offset_idx < len(y) else 0],
                        mode='markers',
                        name='Точка максимума',
                        marker=dict(color='red', size=10, symbol='diamond')
                    ))
                    
                    fig_eg.update_layout(
                        xaxis_title="hv (eV)",
                        yaxis_title=f"(F·hv)^{degree}",
                        title=f"Определение Eg для {selected_sample}",
                        legend=dict(yanchor="top", xanchor="right")
                    )
                    
                    st.plotly_chart(fig_eg, theme="streamlit", width='stretch')
                    
                    # Таблица итоговых результатов по всем образцам
                    st.subheader("Результаты для всех образцов")
                    
                    all_results = []
                    for sample in samples:
                        sample_hv = OpticEg["hv"].values
                        sample_y = OpticEg[sample].values
                        
                        # Для каждого образца находим свои максимумы производной
                        sample_maxima, _ = find_derivative_maxima(sample_hv, sample_y)
                        
                        # Рассчитываем Eg и R² для каждого максимума
                        for m in sample_maxima:
                            fit = fit_linear_region(sample_hv, sample_y, m['index'], window_size=10)
                            if fit:
                                m['fit'] = fit
                                m['Eg'] = fit['Eg']
                                m['r_squared'] = fit['r_squared']
                            else:
                                m['fit'] = None
                                m['Eg'] = np.nan
                                m['r_squared'] = np.nan
                        
                        # Фильтруем максимумы с валидными результатами
                        sample_valid_maxima = [m for m in sample_maxima if m.get('fit') is not None]
                        
                        if sample_valid_maxima:
                            # Сортируем по R² (по убыванию)
                            sample_valid_maxima.sort(key=lambda x: x['r_squared'], reverse=True)
                            
                            # Получаем индивидуальные настройки для данного образца
                            sample_settings = st.session_state.eg_settings.get(sample, {'maxima_idx': 0, 'offset': 0.0})
                            sample_max_idx = min(sample_settings['maxima_idx'], len(sample_valid_maxima) - 1)
                            sample_offset = sample_settings['offset']
                            
                            sample_selected_max = sample_valid_maxima[sample_max_idx]
                            
                            # Применяем индивидуальное смещение для данного образца
                            sample_offset_hv = sample_selected_max['hv'] + sample_offset
                            sample_offset_idx = np.argmin(np.abs(sample_hv - sample_offset_hv))
                            sample_fit = fit_linear_region(sample_hv, sample_y, sample_offset_idx, window_size=10)
                            
                            if sample_fit and not np.isnan(sample_fit['Eg']):
                                eg_val = sample_fit['Eg']
                                r2_val = sample_fit['r_squared']
                            else:
                                eg_val = np.nan
                                r2_val = np.nan
                        else:
                            eg_val = np.nan
                            r2_val = np.nan
                        
                        all_results.append({
                            'Образец': sample,
                            'Eg (eV)': f"{eg_val:.3f}" if not np.isnan(eg_val) else 'N/A',
                            'R²': f"{r2_val:.4f}" if not np.isnan(r2_val) else 'N/A'
                        })
                    
                    results_all_df = pd.DataFrame(all_results)
                    st.table(results_all_df)
                    
                    # Кнопка скачивания результатов
                    csv_results = convert_df(results_all_df)
                    st.download_button(
                        label="Скачать результаты Eg",
                        data=csv_results,
                        file_name='Eg_results.csv',
                        mime='text/csv',
                        type="secondary"
                    )
                    
                else:
                    st.warning("Не удалось найти подходящие максимумы производной для расчёта Eg.")
            else:
                st.warning("Максимумы производной не найдены. Проверьте данные.")
# %%
