import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    "Temperature": [20, 20, 40, 40, 60, 60, 80, 80],
    "Equivalents": [1, 2, 1, 2, 1, 2, 1, 2],
    "Hydralazine (%)": [30, 20, 25, 15, 20, 10, 15, 5],
    "Target Product (%)": [50, 70, 55, 75, 60, 80, 65, 85],
}

df = pd.DataFrame(data)

# Подготовка данных
X = df[["Temperature", "Equivalents"]]
y = df["Target Product (%)"]

# Линейная регрессия
model = LinearRegression()
model.fit(X, y)

# Вывод результатов
print("Коэффициенты регрессии:", model.coef_)
print("Свободный член (интерцепт):", model.intercept_)

# Предсказание
X_new = np.array([[30, 1.5], [70, 2]])  # Новые условия
y_pred = model.predict(X_new)
print("Предсказания для новых условий:", y_pred)

# Визуализация
plt.scatter(df["Temperature"], df["Target Product (%)"], color="blue", label="Целевой продукт")
plt.xlabel("Температура")
plt.ylabel("Целевой продукт (%)")
plt.legend()
plt.show()