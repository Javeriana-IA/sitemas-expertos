import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Cargar los datos
# Reemplaza 'ruta_a_tus_datos.csv' con la ruta a tu archivo de datos
datos = pd.read_csv('dataset.csv', parse_dates=['fecha'])
datos.sort_values('fecha', inplace=True)

# Ingeniería de características: extraer características de la fecha
datos['Año'] = datos['fecha'].dt.year
datos['Mes'] = datos['fecha'].dt.month
datos['Dia'] = datos['fecha'].dt.day

# Descartar la columna original de Fecha
datos.drop('fecha', axis=1, inplace=True)

# Dividir los datos en características (X) y objetivo (y)
X = datos.drop('precio', axis=1)  # Asume que 'Valor_Oro' es tu columna objetivo
y = datos['precio']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Gradient Boosting
modelo = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Error Cuadrático Medio: {mse}")

# Realizar predicciones (reemplazar 'datos_nuevos' con tus datos nuevos)
# predicciones = modelo.predict(datos_nuevos)