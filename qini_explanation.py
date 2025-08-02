import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загружаем данные
discount = pd.read_csv("discountuplift.csv", sep="\t")

def explain_qini_n():
    """
    Объяснение, что такое n в Qini кривой
    """
    print("=== Объяснение переменной n в Qini кривой ===\n")
    
    # 1. Сортируем данные по uplift_score
    ranked = discount.sort_values(by='uplift_score', ascending=False).reset_index(drop=True)
    
    print("1. После сортировки по uplift_score:")
    print(f"   - Всего клиентов: {len(ranked)}")
    print(f"   - Первые 5 клиентов с наивысшим uplift_score:")
    print(ranked[['uplift_score', 'target_class']].head())
    
    # 2. Что такое n
    print("\n2. Что такое n:")
    print("   n - это количество клиентов, на которых мы нацеливаемся")
    print("   n = 0 означает, что мы не выбираем никого")
    print("   n = 100 означает, что мы выбираем топ-100 клиентов")
    print("   n = len(ranked) означает, что мы выбираем всех клиентов")
    
    # 3. Показываем примеры
    print("\n3. Примеры значений n:")
    for n in [0, 10, 50, 100, 200]:
        if n <= len(ranked):
            selected = ranked.head(n)
            print(f"   n = {n}: выбираем топ-{n} клиентов")
            print(f"      Средний uplift_score: {selected['uplift_score'].mean():.4f}")
    
    # 4. Создаем n для графика
    ranked['n'] = ranked.index
    print(f"\n4. Создаем столбец n:")
    print(f"   n принимает значения от 0 до {len(ranked)-1}")
    print(f"   Первые значения n: {ranked['n'].head().tolist()}")
    
    # 5. Рассчитываем uplift для разных n
    N_c = sum(ranked['target_class'] <= 1)  # контрольная группа
    N_t = sum(ranked['target_class'] >= 2)  # тестовая группа
    
    # Создаем индикаторы
    ranked['n_c1'] = 0
    ranked['n_t1'] = 0
    ranked.loc[ranked.target_class == 1, 'n_c1'] = 1  # CR
    ranked.loc[ranked.target_class == 3, 'n_t1'] = 1  # TR
    
    # Кумулятивные доли
    ranked['n_c1/nc'] = ranked['n_c1'].cumsum() / N_c
    ranked['n_t1/nt'] = ranked['n_t1'].cumsum() / N_t
    
    # Uplift для каждого n
    ranked['uplift'] = (ranked['n_t1/nt'] - ranked['n_c1/nc']) * ranked.index
    
    print(f"\n5. Uplift для разных n:")
    for n in [0, 10, 50, 100, 200]:
        if n < len(ranked):
            uplift_at_n = ranked.loc[n, 'uplift']
            print(f"   n = {n}: uplift = {uplift_at_n:.4f}")
    
    # 6. Строим график
    plt.figure(figsize=(12, 8))
    plt.plot(ranked['n'], ranked['uplift'], color='blue', linewidth=2, label='Model Uplift')
    plt.xlabel('Number of customers targeted (n)')
    plt.ylabel('Cumulative Uplift')
    plt.title('Qini Curve - Uplift vs Number of Customers Targeted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\n6. График показывает:")
    print(f"   - По оси X: количество клиентов, на которых нацеливаемся (n)")
    print(f"   - По оси Y: кумулятивный uplift")
    print(f"   - Чем выше кривая, тем лучше модель")

# Запускаем объяснение
explain_qini_n() 