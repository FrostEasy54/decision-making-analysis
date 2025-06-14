import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import f as f_dist

from utils.dispersion_analysis import get_groups_from_df, perform_dispers_analysis
from utils.dynamic_programming import compute_replacement_plan, extract_rs_from_df
from utils.risk_analysis import build_risk_matrix, calculate_danger

# Настройка страницы
st.set_page_config(page_title="Моделирование и анализ принятия решений", layout="wide")
st.title("Моделирование и анализ принятия решений")

# Создаем три вкладки
tab1, tab2, tab3 = st.tabs(
    ["Дисперсионный анализ", "Анализ рисков", "Динамическое программирование"]
)

# Вкладка 1: Дисперсионный анализ
with tab1:
    st.header("Дисперсионный анализ")
    uploaded_file1 = st.file_uploader(
        "Загрузите Excel файл для дисперсионного анализа",
        type=["xlsx", "xls"],
        key="upload_dispers",
    )
    if uploaded_file1 is not None:
        try:
            df1: pd.DataFrame = pd.read_excel(uploaded_file1)
            st.dataframe(df1)

            # Подготовка данных и расчет dispers
            groups = get_groups_from_df(df1)
            if len(groups) >= 2:
                result = perform_dispers_analysis(groups)

                # Отображение результатов
                st.subheader("Результаты дисперсионного анализа:")
                st.write(f"Количество групп: {result['k']}")
                st.write(f"Общее количество наблюдений: {result['N']}")
                st.write(f"Общее среднее: {result['total_mean']:.4f}")

                # Средние по группам
                df_means = pd.DataFrame(
                    {
                        "Группа": df1.columns.to_list(),
                        "Среднее": [round(m, 4) for m in result["group_means"]],
                        "n": result["n_list"],
                    }
                )
                st.table(df_means)

                # Таблица dispers
                df_dispers = pd.DataFrame(
                    {
                        "Источник": ["Между", "Внутри", "Общая"],
                        "Сумма квадратов": [
                            result["SS_between"],
                            result["SS_within"],
                            result["SS_total"],
                        ],
                        "Число степеней свободы": [
                            result["df_between"],
                            result["df_within"],
                            result["df_total"],
                        ],
                        "Дисперсия": [
                            result["MS_between"],
                            result["MS_within"],
                            result["MS_total"],
                        ],
                        "Выборочное значение статистики Фишера": [
                            result["F"],
                            None,
                            None,
                        ],
                    }
                )
                st.table(df_dispers)

                st.write(f"F-статистика: {result['F']:.4f}")
                st.write(f"p-значение: {result['p_value']:.4f}")

                if result["p_value"] < 0.05:
                    st.success(
                        "Есть статистически значимые различия между группами (p < 0.05)."
                    )
                else:
                    st.info(
                        "Нет статистически значимых различий между группами (p ≥ 0.05)."
                    )

                # Визуализация распределения F
                x = np.linspace(
                    0,
                    max(
                        result["F"] * 1.5,
                        f_dist.ppf(0.95, result["df_between"], result["df_within"]),  # type: ignore
                    ),
                    200,
                )
                y = f_dist.pdf(x, result["df_between"], result["df_within"])

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode="lines", name="F-распределение")
                )

                # Добавляем маркер и вертикальную линию для фактического значения F
                f_value = result["F"]
                f_pdf_at_value = f_dist.pdf(
                    f_value, result["df_between"], result["df_within"]
                )
                fig.add_vline(
                    x=f_value,
                    line=dict(dash="dash"),
                    annotation_text="F-статистика",
                    annotation_position="top right",
                )
                fig.add_trace(
                    go.Scatter(
                        x=[f_value],
                        y=[f_pdf_at_value],
                        mode="markers",
                        marker=dict(size=10, symbol="x", color="red"),
                        name="F значение",
                    )
                )

                fig.update_layout(
                    title="Распределение F-статистики",
                    xaxis_title="F",
                    yaxis_title="Плотность",
                )
                st.plotly_chart(fig)
            else:
                st.warning(
                    "Недостаточно групп для проведения дисперсионного анализа (необходимо минимум две группы)."
                )

        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

# Вкладка 2: Анализ рисков
with tab2:
    st.header("Анализ рисков")
    uploaded_file2 = st.file_uploader(
        "Загрузите Excel файл для анализа рисков",
        type=["xlsx", "xls"],
        key="upload_risk"
    )
    if uploaded_file2 is not None:
        try:
            df2: pd.DataFrame = pd.read_excel(uploaded_file2)
            st.subheader("Исходные данные")
            st.dataframe(df2)

            # Вычисление уровня опасности
            df2_danger: pd.DataFrame = calculate_danger(df2)
            st.subheader("Данные с уровнем опасности")
            st.dataframe(df2_danger)

            # Построение карты рисков
            try:
                fig_risk = build_risk_matrix(df2_danger)
                st.plotly_chart(fig_risk)
            except ValueError as e:
                st.warning(str(e))

        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

# Вкладка 3: Динамическое программирование
with tab3:
    st.header("Динамическое программирование")
    uploaded_file3 = st.file_uploader(
        "Загрузите Excel файл для динамического программирования",
        type=["xlsx", "xls"],
        key="upload_dp"
    )
    if uploaded_file3 is not None:
        try:
            df3: pd.DataFrame = pd.read_excel(uploaded_file3, header=None)
            st.subheader("Исходные данные (две строки: прибыль r(t) и остаточная стоимость s(t))")
            st.dataframe(df3)

            # Извлечение r и s
            r, s = extract_rs_from_df(df3)

            # Ввод параметров
            st.subheader("Параметры модели:")
            P = st.number_input("Стоимость перехода на новое графическое ядро (P)", value=7.0, min_value=0.0)
            t0 = st.number_input(
                "Начальный возраст графического ядра (t0)",
                value=1,
                min_value=0,
                max_value=len(r)-1,
                step=1
            )
            if st.button("Рассчитать план", key="calc_dp"):
                try:
                    result_dp = compute_replacement_plan(r, s, P, t0)
                    st.subheader("Максимальная прибыль")
                    st.write(f"**{result_dp['total_profit']:.2f}**")

                    st.subheader("Оптимальный план замены графического ядра")
                    df_plan = pd.DataFrame(result_dp['plan'])
                    st.table(df_plan)

                    # st.subheader("Таблица Беллмана (F)")
                    # df_F = pd.DataFrame(result_dp['F'])
                    # st.dataframe(df_F)
                except Exception as e:
                    st.error(f"Ошибка при расчете: {e}")
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")