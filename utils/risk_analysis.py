import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px


def calculate_danger(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame столбец 'Опасность' = Ущерб * Вероятность для валидных строк.
    Ожидаемые колонки: 'Код', 'Ущерб', 'Вероятность'.
    Значения 1 <= Ущерб, Вероятность <= 10.
    """
    df = df.copy()
    # Приводим к числу и фильтруем
    df["Ущерб"] = pd.to_numeric(df["Ущерб"], errors="coerce")
    df["Вероятность"] = pd.to_numeric(df["Вероятность"], errors="coerce")
    # Убираем название
    df = df.drop("Название риска", axis=1)
    # Вычисляем Danger
    df["Опасность"] = df.apply(
        lambda row: row["Ущерб"] * row["Вероятность"]
        if 1 <= row["Ущерб"] <= 10 and 1 <= row["Вероятность"] <= 10
        else np.nan,
        axis=1,
    )
    return df


def build_risk_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Строит матрицу рисков (5x5) на основе колонок 'Код', 'Ущерб', 'Вероятность'.
    Каждая ячейка содержит перечень кодов рисков и отображает средний уровень опасности.
    Возвращает Plotly Figure.
    """
    # Собираем валидные риски
    risks = df.dropna(subset=["Код", "Ущерб", "Вероятность"])
    risks = risks[
        (risks["Ущерб"].between(1, 10)) & (risks["Вероятность"].between(1, 10))
    ]
    if risks.empty:
        raise ValueError("Нет валидных данных для построения матрицы рисков.")

    # Сетка уровней ущерба и вероятности
    damage_vals = np.arange(1, 11)
    prob_vals = np.arange(1, 11)
    damage_grid, prob_grid = np.meshgrid(damage_vals, prob_vals)
    risk_levels = damage_grid * prob_grid

    # Среднее в блоках 2x2 => матрица 5x5
    avg_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            block = risk_levels[i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2]
            avg_matrix[i, j] = np.mean(block)

    # Группируем коды по блокам
    cell_labels = defaultdict(list)
    for _, row in risks.iterrows():
        x_bin = (int(row["Ущерб"]) - 1) // 2
        y_bin = (int(row["Вероятность"]) - 1) // 2
        cell_labels[(y_bin, x_bin)].append(str(row["Код"]))  # y, x order for heatmap

    # Создаем текст для ячеек
    text_matrix = [["" for _ in range(5)] for _ in range(5)]
    for (y, x), codes in cell_labels.items():
        text_matrix[y][x] = "<br>".join(codes)

    # Построение heatmap
    # Используем цветовую шкалу RdYlGn, но перевернутую
    colors = px.colors.diverging.RdYlGn[::-1]
    fig = go.Figure(
        data=go.Heatmap(
            z=avg_matrix,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=12),
            hoverinfo="text+z",
            x=[f"{i * 2 + 1}–{i * 2 + 2}" for i in range(5)],
            y=[f"{i * 2 + 1}–{i * 2 + 2}" for i in range(5)][::-1],
            colorscale=colors,
            zmin=0,
            zmax=100,
            showscale=True,
            colorbar=dict(title=dict(text="Уровень опасности", side="right")),
        )
    )

    fig.update_layout(
        title="Карта рисков",
        xaxis=dict(
            title="Ущерб",
            tickmode="array",
            tickvals=list(range(5)),
            ticktext=[f"{i * 2 + 1}–{i * 2 + 2}" for i in range(5)], 
        ),
        yaxis=dict(
            title="Вероятность",
            tickmode="array",
            tickvals=list(range(5)),
            ticktext=[f"{i * 2 + 1}–{i * 2 + 2}" for i in range(5)][::-1],
        ),
        width=700,
        height=700,
    )

    return fig
