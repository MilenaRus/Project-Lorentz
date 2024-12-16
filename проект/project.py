import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Лоренцева функция с учетом смещения (offset)
def lorentzian(x, x0, gamma, A, offset):
    return A * (gamma**2) / ((x - x0)**2 + gamma**2) + offset

# 1. Считываем данные из Excel
file_path = "data1.xlsx"
data = pd.read_excel(file_path)

# Предполагаем, что первый столбец - это x, второй - y
x = data.iloc[:, 0]
y = data.iloc[:, 1]

# 2. Строим исходный график
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Исходный график', s=10)

# 3. Аппроксимация данных функцией Лоренца с учетом смещения
initial_guess = [x.mean(), (x.max() - x.min()) / 10, y.max(), y.min()]
params, params_covariance = curve_fit(lorentzian, x, y, p0=initial_guess)

x0, gamma, A, offset = params

# 4. Построение аппроксимированного графика
x_fit = np.linspace(min(x), max(x), 1000)
y_fit = lorentzian(x_fit, *params)
plt.plot(x_fit, y_fit, color='red', label='Лоренц')

# 5. Подсчет площади под кривой с использованием метода трапеций
y_curve = lorentzian(x_fit, *params) - offset  # вычитаем смещение
area_under_curve = np.trapz(y_curve, x_fit)  # Интегрируем методом трапеций
print(f"Площадь: {area_under_curve}")


# 6. Вывод параметров аппроксимации
FWHM_lorentzian = 2 * gamma  # Full Width at Half Maximum для Лоренцевой функции
print(f"Lorentzian: Центр (x0): {x0}, FWHM: {FWHM_lorentzian}, Амплитуда: {A}, Смещение: {offset}")
H=2*area_under_curve/(np.pi*FWHM_lorentzian)
print(f"H: {H}")
# Оформление графика
plt.title("Аппроксимация функцией Лоренца")
plt.xlabel("Магнитное поле")
plt.ylabel("Поглощение")
plt.legend()
plt.grid()
plt.show()