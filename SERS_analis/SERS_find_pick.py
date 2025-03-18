# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import SERS_analis.als as als
import streamlit as st


@st.cache_data()
def data_read(path):
    df = pd.read_table(path,
                       header=None, decimal=",").drop([0, 1], axis=1).T
    columns = ["Raman shift, cm^-1"]
    for i in range(df.shape[1] - 1):
        columns.append(f"Point {i+1}")
    df.columns = columns
    df.index = df[df.columns[0]]
    df.drop(df.columns[0], axis=1, inplace=True)
    df = df.loc[400:]
    return df


@st.cache_data()
def find_inflection_point(df_column, threshold=0):
    values = df_column.tolist()
    inflection_index = None
    inflection_value = None

    for i in range(1, len(values)-1):
        prev = values[i-1]
        current = values[i]
        next_val = values[i+1]

        if (current - prev > threshold) and (current - next_val > threshold):
            inflection_index = df_column.index[i]
            inflection_value = current
            break

    return inflection_index, inflection_value


@st.cache_data()
def baseline(df, lam=1e7, itermax=20):
    df_baseline = df.copy()
    df_line = df.copy()
    for i in range(df.shape[1]):
        X = df[df.columns[i]]
        als_res = als.als(X.values, lam=1e7, itermax=20)
        df_baseline[df.columns[i]] = als_res
        res = X.values - als_res
        df_line[df.columns[i]] = res
    return df_line, df_baseline


@st.cache_data()
def get_inflection_points(df, peaks_points):

    columns = [elem for i in range(1, len(peaks_points)+1)
               for elem in (f"X{i}", f"Y{i}")]
    res_df = pd.DataFrame(columns=columns, index=[
                          f"Point {i+1}" for i in range(df.shape[1])])

    for i in range(df.shape[1]):
        peak_res = []
        for j, peak_point in enumerate(peaks_points):
            X = df[df.columns[i]]
            if j == 0:
                X.loc[1170: 1188].to_csv("test.csv")

            x1, y1 = find_inflection_point(
                X.loc[peak_point[0]: peak_point[1]])

            peak_res.append(x1)
            peak_res.append(y1)

        res_df.loc[f"Point {i+1}"] = peak_res
        print("\n")
    return res_df


def plot_region(df):
    for i in range(df.shape[1]):
        X = df.columns[i]
        fig = plt.figure(figsize=(10, 6))

        plt.subplot(3, 1, 1)
        plt.plot(df[X][1100:1250], marker='o')
        plt.axvspan(1170, 1188, color='pink', alpha=0.8)
        x, y = find_inflection_point(df[X].loc[1170: 1188])
        print(x, y)
        plt.scatter(x, y, color='red', s=100)

        plt.subplot(3, 1, 2)
        plt.plot(df[X][1300:1450], marker='o')
        plt.axvspan(1360, 1378, color='green', alpha=0.4)
        x, y = find_inflection_point(df[X].loc[1360: 1378])
        print(x, y)
        plt.scatter(x, y, color='red', s=100)

        plt.subplot(3, 1, 3)
        plt.plot(df[X][1550:1650], marker='o')
        plt.axvspan(1610, 1625, color='blue', alpha=0.4)
        x, y = find_inflection_point(df[X].loc[1610: 1625])
        print(x, y)
        plt.scatter(x, y, color='red', s=100)
        plt.show()


# %%
if __name__ == "__main__":

    plt.rcParams.update({
        'legend.loc': 'lower center',
        'font.family': 'serif',          # Шрифт с засечками (Times New Roman)
        'font.size': 14,                 # Размер шрифта
        # Размер холста (ширина, высота в дюймах)
        'figure.figsize': (8, 6),
        'figure.dpi': 300,               # Разрешение
        'axes.titlesize': 14,            # Размер заголовка
        'axes.labelsize': 14,            # Размер подписей осей
        'axes.grid': True,               # Включить сетку
        'grid.linewidth': 0.5,           # Толщина линий сетки
        'lines.linewidth': 1.5,          # Толщина линий графиков
        'lines.markersize': 8,           # Размер маркеров
        'xtick.labelsize': 12,           # Размер подписей делений на оси X
        'ytick.labelsize': 12,           # Размер подписей делений на оси Y
        'legend.fontsize': 12,           # Размер шрифта легенды
        'legend.frameon': False,         # Убрать рамку легенды
        'legend.loc': 'right',     # Положение легенды
        'mathtext.fontset': 'stix',      # Стиль математических символов
    })

    samples_name = [r"data\example1\зона 1 - т1 - мап.txt",
                    r"data\example1\зона 1 - т2 - мап.txt",
                    r"data\example1\зона 1 - т3 - мап.txt",
                    r"data\example1\зона 1 - т4 - мап.txt",
                    r"data\example1\зона 1 - т5 - мап.txt"]

    # %%
    samples = []
    for i, sample_name in enumerate(samples_name):
        print(f"Sample {i+1}: {sample_name}")
        df = data_read(sample_name)
        df, _ = baseline(df, lam=1e7, itermax=20)
        samples.append(df)

    # %%
    df_result = pd.DataFrame(index=["Y1_mean", "Y1_std",
                                    "Y2_mean", "Y2_std",
                                    "Y3_mean", "Y3_std",])
    for i, sample in enumerate(samples):
        res_df = get_inflection_points(sample)
        res = [res_df["Y1"].mean(), res_df["Y1"].std(),
               res_df["Y2"].mean(), res_df["Y2"].std(),
               res_df["Y3"].mean(), res_df["Y3"].std()]
        print(f"T{i+1}")
        df_result[f"T{i+1}"] = res

    # %%
    df = df_result.T
    fig = plt.figure(figsize=(12, 8))
    # Параметры для каждого графика

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.errorbar(df.index, df[f"Y{i+1}_mean"],
                    yerr=df[f"Y{i+1}_std"], fmt="o", capsize=10)
        ax.set_title(f"Пик {i+1}", fontsize=12)

    plt.tight_layout()
    plt.show()

    # %%
    plt.errorbar(
        df.index,
        df[f"Y1_mean"],
        yerr=df["Y1_std"],
        fmt="o",
        capsize=10,
        label="Пик 1"
    )
    plt.errorbar(
        df.index,
        df[f"Y2_mean"],
        yerr=df["Y2_std"],
        fmt="o",
        capsize=10,
        label="Пик 2"
    )
    plt.errorbar(
        df.index,
        df[f"Y3_mean"],
        yerr=df["Y3_std"],
        fmt="o",
        capsize=10,
        label="Пик 3"
    )
    plt.legend()
