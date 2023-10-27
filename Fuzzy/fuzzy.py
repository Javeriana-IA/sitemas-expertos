pip install -U scikit-fuzzy


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

# Cargar el dataset
data = pd.read_csv('dataset.csv')

# Extraer precios
precios = data['precio']

# Crear las variables antecedentes y consecuentes
precio = ctrl.Antecedent(np.arange(0, max(precios)+1, 1), 'precio')
prediccion = ctrl.Consequent(np.arange(0, max(precios)+1, 1), 'prediccion')

# Generar funciones de membresía difusa
precio['bajo'] = fuzz.trimf(precio.universe, [0, 0, max(precios)/2])
precio['medio'] = fuzz.trimf(precio.universe, [0, max(precios)/2, max(precios)])
precio['alto'] = fuzz.trimf(precio.universe, [max(precios)/2, max(precios), max(precios)])

prediccion['bajo'] = fuzz.trimf(prediccion.universe, [0, 0, max(precios)/2])
prediccion['medio'] = fuzz.trimf(prediccion.universe, [0, max(precios)/2, max(precios)])
prediccion['alto'] = fuzz.trimf(prediccion.universe, [max(precios)/2, max(precios), max(precios)])

# Crear reglas difusas
rule1 = ctrl.Rule(precio['bajo'], prediccion['bajo'])
rule2 = ctrl.Rule(precio['medio'], prediccion['medio'])
rule3 = ctrl.Rule(precio['alto'], prediccion['alto'])

# Crear y simular un sistema de control difuso
prediccion_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
prediccion_simulador = ctrl.ControlSystemSimulation(prediccion_ctrl)

# Supongamos que el precio actual es 1500
prediccion_simulador.input['precio'] = 1000

# Calcular la predicción
prediccion_simulador.compute()

# Obtener la predicción
print(prediccion_simulador.output['prediccion'])