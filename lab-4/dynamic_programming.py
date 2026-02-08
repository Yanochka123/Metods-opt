from functools import lru_cache
import numpy as np

# ----------------------------
# 1. Исходные данные
# ----------------------------

T = 3
INITIAL_STATE = (100.0, 800.0, 400.0, 600.0)

# Пакеты для управления (1/4 от начальной стоимости)
PACKETS = [25, 200, 100]  # ЦБ1, ЦБ2, Депозит

SCENARIOS = [
    [(0.60, 1.20, 1.10, 1.07),
     (0.30, 1.05, 1.02, 1.03),
     (0.10, 0.80, 0.95, 1.00)],
    
    [(0.30, 1.40, 1.15, 1.01),
     (0.20, 1.05, 1.00, 1.00),
     (0.50, 0.60, 0.90, 1.00)],
    
    [(0.40, 1.15, 1.12, 1.05),
     (0.40, 1.05, 1.01, 1.01),
     (0.20, 0.70, 0.94, 1.00)]
]

# ----------------------------
# 2. Функция дискретизации
# ----------------------------

def discretize_state(s1, s2, s3, cash):
    """Округление до 2 знаков для устойчивости кэширования."""
    return (
        round(s1 / PACKETS[0]) * PACKETS[0],
        round(s2 / PACKETS[1]) * PACKETS[1],
        round(s3 / PACKETS[2]) * PACKETS[2],
        round(cash / 0.01) * 0.01  # кэш округляем до копейки
    )

# ----------------------------
# 3. Улучшенная генерация допустимых действий
# ----------------------------

def get_actions(s1, s2, s3, cash):
    """Генерация допустимых действий с учетом пакетов."""
    actions = []
    
    # Определяем максимальное количество пакетов, которые можно продать
    max_sell1 = int(s1 / PACKETS[0])
    max_sell2 = int(s2 / PACKETS[1])
    max_sell3 = int(s3 / PACKETS[2])
    
    # Определяем максимальное количество пакетов, которые можно купить
    max_buy1 = int(cash / PACKETS[0])
    max_buy2 = int(cash / PACKETS[1])
    max_buy3 = int(cash / PACKETS[2])
    
    # Для упрощения перебора ограничим диапазон
    # Обычно достаточно рассмотреть от -2 до +2 пакетов для каждого актива
    range1 = range(-min(2, max_sell1), min(2, max_buy1) + 1)
    range2 = range(-min(2, max_sell2), min(2, max_buy2) + 1)
    range3 = range(-min(2, max_sell3), min(2, max_buy3) + 1)
    
    for k1 in range1:
        dx1 = k1 * PACKETS[0]
        new_s1 = s1 + dx1
        if new_s1 < 0:
            continue
        
        cash_after1 = cash - dx1 if dx1 > 0 else cash - dx1  # продажа добавляет деньги
        
        for k2 in range2:
            dx2 = k2 * PACKETS[1]
            new_s2 = s2 + dx2
            if new_s2 < 0:
                continue
            
            cash_after2 = cash_after1 - dx2 if dx2 > 0 else cash_after1 - dx2
            
            for k3 in range3:
                dx3 = k3 * PACKETS[2]
                new_s3 = s3 + dx3
                if new_s3 < 0:
                    continue
                
                cash_after3 = cash_after2 - dx3 if dx3 > 0 else cash_after2 - dx3
                
                # Проверяем, что денег хватает (кредит не допускается)
                if cash_after3 >= -1e-6:
                    actions.append((dx1, dx2, dx3))
    
    # Всегда добавляем действие "ничего не делать"
    actions.append((0, 0, 0))
    
    return actions

# ----------------------------
# 4. Рекурсивная функция с мемоизацией (исправленная)
# ----------------------------

policy = {}

@lru_cache(maxsize=None)
def V(t, s1, s2, s3, cash):
    """Возвращает максимальное ожидаемое значение портфеля с этапа t."""
    
    # Базовый случай - конец периода планирования
    if t == T:
        return s1 + s2 + s3 + cash
    
    best_value = -1e18
    best_action = (0, 0, 0)
    
    # Получаем все допустимые действия для текущего состояния
    actions = get_actions(s1, s2, s3, cash)
    
    for dx1, dx2, dx3 in actions:
        # Новое состояние после действия
        ns1 = s1 + dx1
        ns2 = s2 + dx2
        ns3 = s3 + dx3
        ncash = cash - dx1 - dx2 - dx3  # знак уже учтен в dx
        
        # Ожидаемое значение после этого действия
        expected_val = 0.0
        
        # Учитываем все возможные сценарии на текущем этапе
        for prob, m1, m2, m3 in SCENARIOS[t]:
            # Состояние после применения сценария
            next_s1 = ns1 * m1
            next_s2 = ns2 * m2
            next_s3 = ns3 * m3
            
            # Дискретизируем для кэширования
            d1, d2, d3, dc = discretize_state(next_s1, next_s2, next_s3, ncash)
            
            # Рекурсивный вызов для следующего этапа
            future_val = V(t + 1, d1, d2, d3, dc)
            expected_val += prob * future_val
        
        # Выбираем лучшее действие
        if expected_val > best_value:
            best_value = expected_val
            best_action = (dx1, dx2, dx3)
    
    # Сохраняем оптимальное действие
    state_key = (t, s1, s2, s3, cash)
    policy[state_key] = best_action
    
    return best_value

# ----------------------------
# 5. Восстановление оптимальной стратегии (исправленное)
# ----------------------------

def simulate_optimal_strategy():
    s1, s2, s3, cash = INITIAL_STATE
    print("Оптимальная стратегия управления:\n")
    
    total_expected_value = 0
    
    for t in range(T):
        print(f"Этап {t+1}:")
        print(f"  Начальное состояние: ЦБ1={s1:.2f}, ЦБ2={s2:.2f}, Деп={s3:.2f}, Своб.={cash:.2f}")
        
        # Получаем оптимальное действие
        state_key = (t, s1, s2, s3, cash)
        
        if state_key not in policy:
            # Если действие не найдено, вычисляем его
            d1, d2, d3, dc = discretize_state(s1, s2, s3, cash)
            V(0, d1, d2, d3, dc)  # Это заполнит policy
        
        dx1, dx2, dx3 = policy.get(state_key, (0, 0, 0))
        
        # Отображаем действие
        actions_str = []
        if dx1 != 0:
            actions_str.append(f"{'Купить' if dx1 > 0 else 'Продать'} ЦБ1 на {abs(dx1):.0f} д.е.")
        if dx2 != 0:
            actions_str.append(f"{'Купить' if dx2 > 0 else 'Продать'} ЦБ2 на {abs(dx2):.0f} д.е.")
        if dx3 != 0:
            actions_str.append(f"{'Купить' if dx3 > 0 else 'Продать'} Деп на {abs(dx3):.0f} д.е.")
        if not actions_str:
            actions_str = ["Ничего не делать"]
        print("  Управление:", ", ".join(actions_str))
        
        # Применяем управление
        s1 += dx1
        s2 += dx2
        s3 += dx3
        cash -= (dx1 + dx2 + dx3)
        
        # Вычисляем ожидаемое состояние после этапа
        expected_s1, expected_s2, expected_s3 = s1, s2, s3
        
        # Учитываем все сценарии для вычисления ожидаемых значений
        expected_next_s1 = 0
        expected_next_s2 = 0
        expected_next_s3 = 0
        
        for prob, m1, m2, m3 in SCENARIOS[t]:
            expected_next_s1 += prob * s1 * m1
            expected_next_s2 += prob * s2 * m2
            expected_next_s3 += prob * s3 * m3
        
        print(f"  Ожидаемое состояние после этапа: ЦБ1={expected_next_s1:.2f}, ЦБ2={expected_next_s2:.2f}, Деп={expected_next_s3:.2f}, Своб.={cash:.2f}\n")
        
        # Обновляем для следующей итерации (берем среднее)
        s1, s2, s3 = expected_next_s1, expected_next_s2, expected_next_s3
    
    final_value = s1 + s2 + s3 + cash
    print(f"Ожидаемая итоговая стоимость портфеля: {final_value:.2f} д.е.")

# ----------------------------
# 6. Запуск программы
# ----------------------------

def main():
    # Инициализация
    s1, s2, s3, cash = INITIAL_STATE
    
    # Дискретизация начального состояния
    d1, d2, d3, dc = discretize_state(s1, s2, s3, cash)
    
    # Вычисление оптимального значения
    optimal_value = V(0, d1, d2, d3, dc)
    
    print(f"Максимальное ожидаемое значение портфеля: {optimal_value:.2f} д.е.\n")
    print("=" * 60)
    
    # Симуляция оптимальной стратегии
    simulate_optimal_strategy()

if __name__ == "__main__":
    main()