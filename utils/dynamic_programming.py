import numpy as np
import pandas as pd


def extract_rs_from_df(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Из DataFrame без заголовков (две строки) извлекает два списка:
    r (доходы) и s (остаточная стоимость).
    Ожидается DataFrame с shape=(2, n)
    """
    if df.shape[0] != 2 or df.shape[1] < 2:
        raise ValueError("Неверный формат данных: ожидается 2 строки и минимум 2 столбца.")
    r = [float(x) for x in df.iloc[0].tolist()]
    s = [float(x) for x in df.iloc[1].tolist()]
    return r, s


def compute_replacement_plan(r: list[float], s: list[float], P: float, t0: int) -> dict:
    """
    Алгоритм замены оборудования (динамическое программирование).
    r: список доходов по возрастам (len = max_age+1)
    s: список остаточной стоимости по возрастам (len = max_age+1)
    P: стоимость нового оборудования
    t0: начальный возраст оборудования

    Возвращает dict с ключами:
        total_profit: float
        plan: list[dict] с ключами 'year', 'decision', 'age'
        F: np.ndarray (таблица Беллмана)
        decisions: np.ndarray (матрица решений)
    """
    max_age = len(r) - 1
    n_years = len(r) - 1

    # Инициализация
    F = np.zeros((n_years + 1, max_age + 1))
    decisions = np.empty((n_years + 1, max_age + 1), dtype=object)

    # k = 1 (последний год)
    for t in range(max_age + 1):
        if t < max_age:
            F[1][t] = r[t]
            decisions[1][t] = "Оставить"
        else:
            F[1][t] = s[t] - P + r[0]
            decisions[1][t] = "Заменить"

    # k = 2..n_years
    for k in range(2, n_years + 1):
        for t in range(max_age + 1):
            # keep
            if t < max_age:
                keep_value = r[t] + F[k-1][t+1]
            else:
                keep_value = -np.inf
            # Заменить
            replace_value = s[t] - P + r[0] + F[k-1][1]
            # choose
            if keep_value >= replace_value:
                F[k][t] = keep_value
                decisions[k][t] = "Оставить"
            else:
                F[k][t] = replace_value
                decisions[k][t] = "Заменить"

    # Восстановление плана
    plan = []
    current_age = t0
    total_profit = F[n_years][current_age]
    for k in range(n_years, 0, -1):
        decision = decisions[k][current_age]
        plan.append({
            'Год': n_years - k + 1,
            'Решение': decision,
            'Возраст': current_age
        })
        if decision == "Заменить":
            current_age = 1
        else:
            current_age += 1

    return {
        'total_profit': total_profit,
        'plan': plan,
        'F': F,
        'decisions': decisions
    }
