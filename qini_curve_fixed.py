import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# загружаем DataFrame
discount = pd.read_csv("discountuplift.csv", sep="\t")

def qini_df(df):
    """
    Функция для расчета Qini Score и построения Qini кривой
    
    Параметры:
    df - DataFrame с данными
    
    Возвращает:
    qini_score - значение Qini Score
    """
    # 1. Отранжируем выборку по значению uplift в порядке убывания
    ranked = df.sort_values(by='uplift_score', ascending=False).reset_index(drop=True)
    
    # 2. Подсчитываем количество наблюдений в контрольной и тестовой группах
    N_c = sum(ranked['target_class'] <= 1)  # контрольная группа (0, 1)
    N_t = sum(ranked['target_class'] >= 2)  # тестовая группа (2, 3)
    
    # 3. Создаем индикаторы для положительных исходов
    ranked['n_c1'] = 0  # количество положительных исходов в контрольной группе
    ranked['n_t1'] = 0  # количество положительных исходов в тестовой группе
    
    # target_class: 0=CN, 1=CR, 2=TN, 3=TR
    # Положительные исходы: CR (1) и TR (3)
    ranked.loc[ranked.target_class == 1, 'n_c1'] = 1  # CR
    ranked.loc[ranked.target_class == 3, 'n_t1'] = 1  # TR
    
    # 4. Рассчитываем кумулятивные доли
    ranked['n_c1/nc'] = ranked['n_c1'].cumsum() / N_c
    ranked['n_t1/nt'] = ranked['n_t1'].cumsum() / N_t
    
    # 5. Рассчитываем Qini кривую
    # Uplift = (доля положительных в тесте - доля положительных в контроле) * количество наблюдений
    ranked['uplift'] = (ranked['n_t1/nt'] - ranked['n_c1/nc']) * ranked.index
    
    # 6. Рассчитываем случайную кривую (baseline)
    # Случайная кривая - это прямая линия от (0,0) до (N, общий uplift)
    total_uplift = (ranked['n_t1'].sum() / N_t - ranked['n_c1'].sum() / N_c) * len(ranked)
    ranked['random_uplift'] = (total_uplift / len(ranked)) * ranked.index
    
    # 7. Рассчитываем оптимальную кривую
    # Оптимальная кривая - это прямая линия до точки максимального uplift, затем горизонтальная
    max_uplift = (ranked['n_t1'].sum() / N_t - ranked['n_c1'].sum() / N_c) * N_t
    ranked['optimum_uplift'] = np.minimum(max_uplift, ranked.index * (max_uplift / N_t))
    
    # 8. Строим график
    plt.figure(figsize=(12, 8))
    plt.plot(ranked.index, ranked['uplift'], color='blue', linewidth=2, label='Model')
    plt.plot(ranked.index, ranked['random_uplift'], color='red', linewidth=2, linestyle='--', label='Random')
    plt.plot(ranked.index, ranked['optimum_uplift'], color='green', linewidth=2, linestyle='--', label='Optimum')
    
    plt.xlabel('Number of customers targeted')
    plt.ylabel('Cumulative Uplift')
    plt.title('Qini Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 9. Рассчитываем Qini Score
    # Qini Score = площадь под моделью - площадь под случайной кривой
    from sklearn.metrics import auc
    
    # Площадь под кривыми
    model_auc = auc(ranked.index, ranked['uplift'])
    random_auc = auc(ranked.index, ranked['random_uplift'])
    
    qini_score = model_auc - random_auc
    
    print(f"Qini Score: {qini_score:.4f}")
    print(f"Model AUC: {model_auc:.4f}")
    print(f"Random AUC: {random_auc:.4f}")
    
    # Дополнительная статистика
    print(f"\nСтатистика:")
    print(f"Всего наблюдений: {len(ranked)}")
    print(f"Контрольная группа: {N_c}")
    print(f"Тестовая группа: {N_t}")
    print(f"Положительных исходов в контроле: {ranked['n_c1'].sum()}")
    print(f"Положительных исходов в тесте: {ranked['n_t1'].sum()}")
    
    return qini_score

# Запускаем функцию
qini_score = qini_df(discount) 