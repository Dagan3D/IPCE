# pages/SEM.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.filters import frangi

try:
    import porespy as ps
    HAS_PORESPY = True
except ImportError:
    HAS_PORESPY = False

from sem_utils import SEMImage, analyze_fibers

st.set_page_config(page_title="SEM анализ", layout="wide")
st.title("📷 Анализ SEM-изображений")

def apply_stage(img, stage_name, params, original_for_overlay=None, skeleton_img=None):
    """Применяет этап обработки к изображению"""
    if stage_name == "Исходное":
        crop = params.get('crop_rect', None)
        if crop is not None:
            x1, y1, x2, y2 = crop
            return img[y1:y2, x1:x2]
        return img.copy()
    
    elif stage_name == "Обрезка":
        return img
    
    elif stage_name == "Яркость/контраст":
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 0)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    elif stage_name == "Фильтр Франги":
        sigmas = range(params['sigmas_start'], params['sigmas_end'], params['sigmas_step'])
        frangi_img = frangi(img, sigmas=list(sigmas), black_ridges=False)
        frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return frangi_norm
    
    elif stage_name == "Бинаризация":
        method = params.get('method', 'otsu')
        if method == 'otsu':
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = params.get('thresh', 128)
            _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        return binary
    
    elif stage_name == "Морфология (закрытие)":
        kernel_size = params.get('kernel_size', 7)
        fill_holes = params.get('fill_holes', True)
        
        if kernel_size >= 3:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            
        if fill_holes:
            img = binary_fill_holes(img > 0).astype(np.uint8) * 255
            
        return img
    
    elif stage_name == "Скелетизация":
        return (skeletonize(img > 0).astype(np.uint8) * 255)
    
    elif stage_name == "Карта расстояний":
        dist = distance_transform_edt(img)
        dist_norm = (dist / dist.max() * 255).astype(np.uint8)
        return dist_norm
    
    elif stage_name == "Наложение (скелет + оригинал)":
        orig = original_for_overlay
        skel = skeleton_img
        if orig is None or skel is None:
            return img
        overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        skel_color = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        skel_color[:,:,0] = 0  # reddish
        return cv2.addWeighted(overlay, 0.7, skel_color, 0.3, 0)
        
    elif stage_name == "Локальная толщина (PoreSpy)":
        if not HAS_PORESPY:
            st.error("Библиотека Porespy не установлена. Выполните: pip install porespy")
            return np.zeros_like(img, dtype=np.float32)
        # img здесь — это бинарная маска
        lt_map = ps.filters.local_thickness(img > 0)
        return lt_map  # Возвращаем массив float с радиусами!
        
    elif stage_name == "Цветовая карта толщин":
        # img здесь — это float-массив карты толщин
        lt_map = img
        if lt_map.max() > 0:
            norm = (lt_map / lt_map.max() * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(lt_map, dtype=np.uint8)
            
        # Используем красивую тепловую карту (Turbo)
        cmap = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        cmap[lt_map == 0] = [0, 0, 0]  # Фон делаем черным
        return cmap
    
    else:
        return img

# ---- Загрузка ----
uploaded_file = st.file_uploader("Загрузите SEM-изображение", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        st.error("Не удалось прочитать изображение.")
        st.stop()

    pixel_size_nm = st.number_input("Размер пикселя (нм)", value=8.371, step=0.1)
    analysis_type = st.radio("Тип анализа",["Поры", "Волокна"], horizontal=True)

    calc_method = "Скелетизация"
    if analysis_type == "Волокна":
        calc_method = st.radio("Метод расчета волокон",["Скелетизация (EDT)", "Локальная толщина (PoreSpy)"], horizontal=True)
        if calc_method == "Локальная толщина (PoreSpy)" and not HAS_PORESPY:
            st.warning("⚠️ Библиотека Porespy не обнаружена. Расчет будет невозможен. Введите в терминале: pip install porespy")

    st.markdown("---")
    step_mode = st.checkbox("🖼️ Пошаговый просмотр этапов", value=False,
                            help="Включите для детальной настройки каждого этапа")

    if step_mode:
        if analysis_type == "Поры":
            stages =["Исходное", "Яркость/контраст", "Бинаризация", "Морфология (закрытие)", "Результат (поры)"]
        else:
            if calc_method == "Локальная толщина (PoreSpy)":
                stages =["Исходное", "Фильтр Франги", "Бинаризация", "Морфология (закрытие)", "Локальная толщина (PoreSpy)", "Цветовая карта толщин"]
            else:
                stages =["Исходное", "Фильтр Франги", "Бинаризация", "Морфология (закрытие)", "Скелетизация", "Карта расстояний", "Наложение (скелет + оригинал)"]

        # Инициализация session_state
        if "processed_images" not in st.session_state:
            st.session_state.processed_images = {}
            st.session_state.params_snapshot = {}
        if "step_idx" not in st.session_state:
            st.session_state.step_idx = 0
        if "stage_params" not in st.session_state:
            st.session_state.stage_params = {}
        
        # Корректировка индекса при смене метода (если массив этапов стал короче)
        if st.session_state.step_idx >= len(stages):
            st.session_state.step_idx = len(stages) - 1

        if "crop_rect" not in st.session_state:
            st.session_state.crop_rect = (0, 0, img_original.shape[1], img_original.shape[0])

        # ---- Навигация ----
        col1, col2, col3 = st.columns([1, 1, 6])
        with col1:
            if st.button("◀ Назад") and st.session_state.step_idx > 0:
                st.session_state.step_idx -= 1
                st.rerun()
        with col2:
            if st.button("Вперёд ▶") and st.session_state.step_idx < len(stages) - 1:
                st.session_state.step_idx += 1
                st.rerun()
        with col3:
            st.write(f"**Этап {st.session_state.step_idx+1} из {len(stages)}:** {stages[st.session_state.step_idx]}")

        current_stage = stages[st.session_state.step_idx]

        # ---- Параметры текущего этапа ----
        params = st.session_state.stage_params.get(current_stage, {})
        if current_stage == "Исходное":
            st.subheader("Обрезка изображения")
            enable_crop = st.checkbox("Обрезать", value=st.session_state.crop_rect != (0,0,img_original.shape[1],img_original.shape[0]))
            if enable_crop:
                h, w = img_original.shape
                left = st.slider("Левая граница (%)", 0, 100, 0, key="crop_left")
                right = st.slider("Правая граница (%)", 0, 100, 100, key="crop_right")
                top = st.slider("Верхняя граница (%)", 0, 100, 0, key="crop_top")
                bottom = st.slider("Нижняя граница (%)", 0, 100, 100, key="crop_bottom")
                x1, x2 = min(int(w * left / 100), int(w * right / 100)), max(int(w * left / 100), int(w * right / 100))
                y1, y2 = min(int(h * top / 100), int(h * bottom / 100)), max(int(h * top / 100), int(h * bottom / 100))
                if x1 == x2: x2 = w
                if y1 == y2: y2 = h
                st.session_state.crop_rect = (x1, y1, x2, y2)
                params['crop_rect'] = st.session_state.crop_rect
            else:
                st.session_state.crop_rect = (0, 0, img_original.shape[1], img_original.shape[0])
                params['crop_rect'] = None
        else:
            if analysis_type == "Поры":
                if current_stage == "Яркость/контраст":
                    params['alpha'] = st.slider("Контраст", 0.5, 3.0, params.get('alpha', 1.0), 0.05)
                    params['beta'] = st.slider("Яркость", -100, 100, params.get('beta', 0))
                elif current_stage == "Бинаризация":
                    method = st.selectbox("Метод", ["otsu", "manual"], index=0 if params.get('method', 'otsu')=='otsu' else 1)
                    params['method'] = method
                    if method == "manual":
                        params['thresh'] = st.slider("Порог", 0, 255, params.get('thresh', 128))
                elif current_stage == "Морфология (закрытие)":
                    params['kernel_size'] = st.slider("Размер ядра", 3, 21, params.get('kernel_size', 9), step=2)
            else:
                if current_stage == "Фильтр Франги":
                    params['sigmas_start'] = st.number_input("Нач. толщина (px)", 1, 50, params.get('sigmas_start', 5))
                    params['sigmas_end'] = st.number_input("Кон. толщина (px)", 2, 200, params.get('sigmas_end', 40))
                    params['sigmas_step'] = st.number_input("Шаг", 1, 20, params.get('sigmas_step', 2))
                elif current_stage == "Бинаризация":
                    method = st.selectbox("Метод", ["otsu", "manual"], index=0 if params.get('method', 'otsu')=='otsu' else 1)
                    params['method'] = method
                    if method == "manual":
                        params['thresh'] = st.slider("Порог", 0, 255, params.get('thresh', 128))
                elif current_stage == "Морфология (закрытие)":
                    params['kernel_size'] = st.slider("Размер ядра", 1, 31, params.get('kernel_size', 7), step=2)
                    params['fill_holes'] = st.checkbox("Автоматически заливать замкнутые пустоты", value=params.get('fill_holes', True))

        st.session_state.stage_params[current_stage] = params

        # ---- Умное кэширование и получение изображения ----
        def get_image_for_step(step_idx, stages, orig_img):
            stale_from = None
            for i in range(step_idx + 1):
                stage = stages[i]
                frozen = st.session_state.params_snapshot.get(stage)
                current = st.session_state.stage_params.get(stage, {})
                if frozen != current:
                    stale_from = i
                    for j in range(i, len(stages)):
                        st.session_state.processed_images.pop(stages[j], None)
                    break

            if stages[step_idx] in st.session_state.processed_images and stale_from is None:
                return st.session_state.processed_images[stages[step_idx]]

            start_idx = stale_from if stale_from is not None else 0
            if start_idx > 0 and stages[start_idx-1] in st.session_state.processed_images:
                img = st.session_state.processed_images[stages[start_idx-1]].copy()
            else:
                img = orig_img.copy()

            binary_img = None
            skeleton_img = None
            original_for_overlay = None

            if "Бинаризация" in st.session_state.processed_images:
                binary_img = st.session_state.processed_images["Бинаризация"].copy()
            if "Морфология (закрытие)" in st.session_state.processed_images:
                binary_img = st.session_state.processed_images["Морфология (закрытие)"].copy()
            if "Скелетизация" in st.session_state.processed_images:
                skeleton_img = st.session_state.processed_images["Скелетизация"].copy()
            if "Исходное" in st.session_state.processed_images:
                original_for_overlay = st.session_state.processed_images["Исходное"].copy()

            for i in range(start_idx, step_idx + 1):
                stage = stages[i]
                stage_params = st.session_state.stage_params.get(stage, {})

                if stage in ["Карта расстояний", "Локальная толщина (PoreSpy)"]:
                    img_to_use = binary_img if binary_img is not None else img
                    img = apply_stage(img_to_use, stage, stage_params)
                elif stage == "Наложение (скелет + оригинал)":
                    img = apply_stage(None, stage, stage_params, original_for_overlay=original_for_overlay, skeleton_img=skeleton_img)
                else:
                    img = apply_stage(img, stage, stage_params, original_for_overlay=original_for_overlay, skeleton_img=skeleton_img)

                if stage in["Бинаризация", "Морфология (закрытие)"]:
                    binary_img = img.copy()
                elif stage == "Скелетизация":
                    skeleton_img = img
                elif stage == "Исходное":
                    original_for_overlay = img.copy()

                st.session_state.processed_images[stage] = img
                st.session_state.params_snapshot[stage] = stage_params.copy()

            return img

        cumulative_img = get_image_for_step(st.session_state.step_idx, stages, img_original)

        # ---- Отображение ----
        disp_img = cumulative_img
        # Если это чистая float-карта, нормализуем только для визуализации, чтобы Streamlit не ругался
        if current_stage == "Локальная толщина (PoreSpy)":
            if disp_img.max() > 0:
                disp_img = (disp_img / disp_img.max() * 255).astype(np.uint8)
            else:
                disp_img = np.zeros_like(disp_img, dtype=np.uint8)

        st.image(disp_img, caption=current_stage, use_container_width=False, width=800, clamp=True)

        # ---- Результаты ----
        if ((analysis_type == "Поры" and current_stage == "Результат (поры)") or
            (analysis_type == "Волокна" and current_stage in ("Карта расстояний", "Наложение (скелет + оригинал)", "Цветовая карта толщин"))):

            st.subheader("Результаты анализа")
            
            if analysis_type == "Поры":
                # [Блок пор остается без изменений]
                min_area_nm2 = st.number_input("Мин. площадь поры (нм²)", 10, 10000, 350)
                max_area_nm2 = st.number_input("Макс. площадь поры (нм²)", 100, 50000, 10000)
                if st.button("Рассчитать поры"):
                    processed_img = cumulative_img.copy()
                    processed_img = cv2.bitwise_not(processed_img)
                    px_area = pixel_size_nm ** 2
                    min_area_px = min_area_nm2 / px_area
                    max_area_px = max_area_nm2 / px_area
                    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    records =[]
                    for cnt in contours:
                        area_px = cv2.contourArea(cnt)
                        if area_px < min_area_px or area_px > max_area_px: continue
                        records.append({
                            'Area_nm2': area_px * px_area,
                            'Diameter_nm': 2 * np.sqrt((area_px * px_area) / np.pi)
                        })
                    df_pores = pd.DataFrame(records)
                    if len(df_pores) == 0:
                        st.warning("Поры не найдены.")
                    else:
                        st.success(f"Найдено пор: {len(df_pores)}")
                        st.dataframe(df_pores.describe())
                        fig, ax = plt.subplots()
                        ax.hist(df_pores['Diameter_nm'], bins=20)
                        ax.set_xlabel("Диаметр (нм)")
                        st.pyplot(fig)
                        st.download_button("Скачать CSV", df_pores.to_csv(sep=';', index=False).encode('utf-8'), "pores.csv", "text/csv")
            
            else:  # Волокна
                min_diam = st.number_input("Мин. диаметр (нм)", 1, 500, 10)
                max_diam = st.number_input("Макс. диаметр (нм)", 10, 2000, 150)
                
                if st.button("🚀 Запустить расчет и построить график", type="primary"):
                    diameters_nm = None
                    is_area_weighted = False
                    
                    # Логика получения данных
                    if calc_method == "Локальная толщина (PoreSpy)":
                        lt_map = st.session_state.processed_images.get("Локальная толщина (PoreSpy)")
                        if lt_map is not None:
                            diameters_nm = lt_map[lt_map > 0] * 2 * pixel_size_nm
                            is_area_weighted = True
                    else:
                        binary_img = st.session_state.processed_images.get("Морфология (закрытие)")
                        if binary_img is None: binary_img = st.session_state.processed_images.get("Бинаризация")
                        skeleton_img = st.session_state.processed_images.get("Скелетизация")
                        
                        if binary_img is not None and skeleton_img is not None:
                            distance_map = distance_transform_edt(binary_img)
                            radii_px = distance_map[skeleton_img > 0]
                            diameters_nm = radii_px * 2 * pixel_size_nm

                    if diameters_nm is not None:
                        mask = (diameters_nm >= min_diam) & (diameters_nm <= max_diam)
                        df_fibers = pd.DataFrame({'diameter_nm': diameters_nm[mask]})
                        
                        if len(df_fibers) == 0:
                            st.warning("Волокна в заданном диапазоне не обнаружены.")
                        else:
                            st.success(f"Анализ завершен! Обработано точек: {len(df_fibers):,}")
                            
                            # --- КРАСИВЫЙ ВЫВОД ---
                            stats = df_fibers['diameter_nm'].describe()
                            
                            # 1. Сводные метрики сверху
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Средний диаметр", f"{stats['mean']:.1f} нм")
                            m2.metric("Медиана (D50)", f"{stats['50%']:.1f} нм")
                            m3.metric("Станд. отклонение", f"{stats['std']:.1f} нм")
                            m4.metric("Макс. диаметр", f"{stats['max']:.1f} нм")
                            
                            st.markdown("---")
                            
                            # 2. Колонки для таблицы и графика
                            col_table, col_chart = st.columns([1, 2])
                            
                            with col_table:
                                st.write("**📊 Подробная статистика**")
                                st.dataframe(stats, use_container_width=True)
                                
                                # Кнопка скачивания рядом с таблицей
                                csv = df_fibers.to_csv(sep=';', index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Скачать данные (.csv)",
                                    data=csv,
                                    file_name="nanowires_stats.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                if is_area_weighted:
                                    st.caption("ℹ️ Метод: Local Thickness. Гистограмма взвешена по площади.")
                                else:
                                    st.caption("ℹ️ Метод: Скелетизация. Расчет по осевым линиям.")

                            with col_chart:
                                import plotly.express as px
                                
                                # Создаем интерактивную гистограмму
                                fig = px.histogram(
                                    df_fibers, 
                                    x="diameter_nm",
                                    nbins=50,
                                    title="Распределение толщин нанонитей",
                                    labels={'diameter_nm': 'Диаметр (нм)', 'count': 'Кол-во (пикс.)'},
                                    color_discrete_sequence=['#2E7D32'], # Глубокий зеленый
                                    marginal="box" # Добавляет "ящик с усами" сверху для наглядности выбросов
                                )
                                
                                fig.update_layout(
                                    hovermode="x unified",
                                    template="plotly_white",
                                    margin=dict(l=20, r=20, t=50, b=20),
                                    bargap=0.05
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Быстрый режим временно отключён. Используйте пошаговый режим.")

else:
    st.info("👈 Загрузите изображение для начала работы.")