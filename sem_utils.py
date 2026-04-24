# sem_utils.py
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.filters import frangi
from typing import Optional, Tuple, Dict, List
import re

class SEMImage:
    """
    Класс для анализа SEM-изображений: поиск пор, расчёт характеристик.
    """
    def __init__(self, img: np.ndarray, pixel_size_nm: Optional[float] = None, 
                 filename: Optional[str] = None):
        self.img = img
        self.filename = filename
        self.pixel_size_nm = pixel_size_nm
        self.pore_table: Optional[pd.DataFrame] = None

        if self.pixel_size_nm is None and filename is not None:
            self.pixel_size_nm = self._extract_pixel_size_from_filename(filename)

    @property
    def size_nm(self) -> Tuple[float, float]:
        if self.img is not None and self.pixel_size_nm is not None:
            return self.img.shape[1] * self.pixel_size_nm, self.img.shape[0] * self.pixel_size_nm
        return (0.0, 0.0)

    @property
    def area_nm2(self) -> float:
        w, h = self.size_nm
        return w * h

    def _extract_pixel_size_from_filename(self, filename: str) -> Optional[float]:
        match = re.search(r'(\d+(?:\.\d+)?)\s*nm', filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def process_pores(self, min_area_nm2: float = 350.0, max_area_nm2: float = 10000.0,
                      add_brightness: int = 250, morph_kernel_size: int = 9) -> pd.DataFrame:
        if self.pixel_size_nm is None:
            raise ValueError("Не задан размер пикселя (pixel_size_nm).")

        px_area = self.pixel_size_nm ** 2
        min_area_px = min_area_nm2 / px_area
        max_area_px = max_area_nm2 / px_area

        img = self.img.copy()
        img = cv2.add(img, add_brightness)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_inv = cv2.bitwise_not(binary)

        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        records =[]
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < min_area_px or area_px > max_area_px:
                continue
            area_nm2 = area_px * px_area
            perimeter_px = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0.0
            diameter_nm = 2 * np.sqrt(area_nm2 / np.pi)

            records.append({
                'Area_nm2': area_nm2,
                'Diameter_nm': diameter_nm,
                'Circularity': circularity
            })

        self.pore_table = pd.DataFrame(records)
        return self.pore_table


def analyze_fibers(image: np.ndarray, pixel_size_nm: float,
                   frangi_sigmas: Tuple[int, int, int] = (5, 40, 2),
                   min_diameter_nm: float = 10.0,
                   max_diameter_nm: float = 200.0,
                   otsu_threshold: bool = True,
                   close_kernel_size: int = 7,
                   fill_holes: bool = True,
                   method: str = 'local_thickness') -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Анализ волокон на SEM-изображении.
    Поддерживаемые методы (method): 'skeleton' или 'local_thickness'
    """
    # 1. Фильтр Франги
    sigmas = range(frangi_sigmas[0], frangi_sigmas[2], frangi_sigmas[1])
    frangi_img = frangi(image, sigmas=list(sigmas), black_ridges=False)
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Бинаризация
    if otsu_threshold:
        _, binary = cv2.threshold(frangi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(frangi_norm, 30, 255, cv2.THRESH_BINARY)

    # 2.5 Морфологическое закрытие
    if close_kernel_size >= 3:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
    # 2.6 Заливка замкнутых пустот
    if fill_holes:
        binary = binary_fill_holes(binary > 0).astype(np.uint8) * 255

    images = {
        'frangi': frangi_norm,
        'binary': binary
    }

    # 3. Извлечение толщины
    if method == 'local_thickness':
        try:
            import porespy as ps
        except ImportError:
            raise ImportError("Для метода 'local_thickness' установите библиотеку porespy (pip install porespy).")
        
        # Строим карту локальной толщины
        lt_map = ps.filters.local_thickness(binary > 0)
        diameters_px = lt_map[lt_map > 0] * 2
        images['local_thickness_map'] = lt_map
        
    else: # классический метод со скелетом
        skeleton = skeletonize(binary > 0).astype(np.uint8) * 255
        distance_map = distance_transform_edt(binary)
        radii_px = distance_map[skeleton > 0]
        diameters_px = radii_px * 2
        
        images['skeleton'] = skeleton
        images['distance_map'] = (distance_map / distance_map.max() * 255).astype(np.uint8)

    # 4. Перевод в нанометры и фильтрация
    diameters_nm = diameters_px * pixel_size_nm
    diameters_nm = diameters_nm[(diameters_nm >= min_diameter_nm) & (diameters_nm <= max_diameter_nm)]

    df = pd.DataFrame({'diameter_nm': diameters_nm})
    return df, images