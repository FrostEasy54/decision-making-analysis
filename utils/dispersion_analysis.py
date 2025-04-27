import numpy as np
from scipy.stats import f
import pandas as pd


def get_groups_from_df(df: pd.DataFrame) -> list[list[float]]:
    """
    Преобразует DataFrame в список групп для dispers_analysis.
    Каждая колонка DataFrame считается отдельной группой, пропуская пустые и нечисловые значения.
    """
    groups = []
    for col in df.columns:
        # Отфильтровать ненулевые и нечисловые значения
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if not series.empty:
            groups.append(series.tolist())
    return groups


def perform_dispers_analysis(data: list[list[float]]) -> dict:
    """
    Выполняет однофакторный дисперсионный анализ (dispers_analysis) для заданных групп.
    Возвращает словарь с результатами расчёта:
    SS, df, MS, F-статистика, p-значение и прочее.
    """
    # Фильтруем пустые группы
    data = [group for group in data if len(group) > 0]
    k = len(data)  # количество групп
    if k < 2:
        raise ValueError("Для dispers_analysis требуется как минимум две группы данных.")

    # Число наблюдений в каждой группе (предполагается одинаковое)
    n_list = [len(group) for group in data]
    # Общие наблюдения
    N = sum(n_list)

    # Вычисляем средние
    group_means = [np.mean(group) for group in data]
    total_mean = np.mean([x for group in data for x in group])

    # Суммы квадратов
    SS_total = sum((x - total_mean) ** 2 for group in data for x in group)
    SS_between = sum(len(group) * (mean - total_mean) ** 2 for group, mean in zip(data, group_means))
    SS_within = SS_total - SS_between

    # Степени свободы
    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    # Средние квадраты
    MS_between = SS_between / df_between
    MS_within = SS_within / df_within
    MS_total = SS_total / df_total

    # F-статистика и p-значение
    F_stat = MS_between / MS_within
    p_value = 1 - f.cdf(F_stat, df_between, df_within)

    return {
        'SS_total': SS_total,
        'SS_between': SS_between,
        'SS_within': SS_within,
        'df_between': df_between,
        'df_within': df_within,
        'df_total': df_total,
        'MS_between': MS_between,
        'MS_within': MS_within,
        'MS_total': MS_total,
        'F': F_stat,
        'p_value': p_value,
        'group_means': group_means,
        'total_mean': total_mean,
        'k': k,
        'n_list': n_list,
        'N': N
    }
