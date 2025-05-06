import numpy as np

def ROMP(y, Phi, K, epsilon=1e-4, max_iter=1000):
    """
    Реализация алгоритма ROMP (Regularized Orthogonal Matching Pursuit) для восстановления разреженных сигналов.

    Аргументы:
        y (np.ndarray): Вектор измерений (N x 1).
        Phi (np.ndarray): Матрица измерений (N x d).
        K (int): Ожидаемый уровень разреженности сигнала.
        epsilon (float): Допуск для остановки (изменение нормы остатка).
        max_iter (int): Максимальное количество итераций.

    Возвращает:
        np.ndarray: Восстановленный разреженный сигнал (d x 1).
    """

    N, d = Phi.shape
    x_recovered = np.zeros(d)
    r = y.copy()  # Остаток инициализируется вектором измерений
    support = set()  # Множество активных индексов (поддержка)

    for _ in range(max_iter):
        # Шаг 1: Вычисление корреляций
        correlations = np.abs(Phi.T @ r)

        # Шаг 2: Выбор кандидатов
        # Находим индексы K наибольших по модулю корреляций
        candidate_indices = np.argsort(correlations)[-K:]

        # Шаг 3: Регуляризация и выбор наилучшего подмножества
        best_subset = set()
        best_error = np.inf

        # Перебираем все подмножества кандидатов (можно использовать itertools.combinations)
        # Для простоты, здесь рассмотрим только добавление всех кандидатов сразу
        current_support = support.union(candidate_indices)

        if not current_support:
            # Если нет активных индексов, выбираем один с наибольшей корреляцией
            best_subset = {np.argmax(correlations)}
        else:
            # Проекция на текущую поддержку
            Phi_subset = Phi[:, list(current_support)]
            try:
                # Решение задачи наименьших квадратов для текущей поддержки
                x_subset, residuals, rank, s = np.linalg.lstsq(Phi_subset, y, rcond=None)
                # Вычисление остатка
                r_new = y - Phi_subset @ x_subset
                current_error = np.linalg.norm(r_new)

                if current_error < best_error:
                    best_error = current_error
                    best_subset = current_support
                    r = r_new
                    support = best_subset
                    # Обновляем восстановленный сигнал
                    x_recovered[list(support)] = x_subset
                    # Обнуляем остальные элементы вне поддержки
                    x_recovered[list(set(range(d)) - support)] = 0

            except np.linalg.LinAlgError:
                # Обработка возможных ошибок при решении СЛАУ (например, сингулярность)
                print("Предупреждение: Ошибка при решении СЛАУ.")
                break

        # Шаг 4: Проверка критерия остановки
        if np.linalg.norm(r) < epsilon:
            break

    return x_recovered

# Пример использования с вашими параметрами
d = 100  # Размерность сигнала
N = 30   # Количество измерений
K = 5   # Уровень разреженности

# Генерация разреженного сигнала
x_true = np.zeros(d)
x_true[np.random.choice(d, K, replace=False)] = np.random.randn(K)

# Генерация случайной матрицы измерений
Phi = np.random.randn(N, d) / np.sqrt(N)

# Получение измерений
y = Phi @ x_true

# Добавление шума (опционально)
noise_level = 0.01
y += noise_level * np.random.randn(N)

# Восстановление сигнала с помощью ROMP
x_recovered = ROMP(y, Phi, K, epsilon=1e-4)

# Проверка точности восстановления
error = np.linalg.norm(x_true - x_recovered)
print(f"Ошибка восстановления: {error:.4f}")