import pandas as pd
import numpy as np
from skrub import TableReport
import matplotlib.pyplot as plt

df = pd.read_csv('data/datos_demograficos_ine.csv', sep=';', encoding='ISO-8859-1')

# Eliminación de columnas sin uso o vacías
df = df.drop(['Municipios', 'Secciones'])

# Mapeo de distritos con nombres
mapeo = {
    '01': 'Centro',
    '02': 'Arganzuela',
    '03': 'Retiro',
    '04': 'Salamanca',
    '05': 'Chamartín',
    '06': 'Tetuán',
    '07': 'Chamberí',
    '08': 'Fuencarral',
    '09': 'Moncloa',
    '10': 'Latina',
    '11': 'Carabanchel',
    '12': 'Usera',
    '13': 'Puente de Vallecas',
    '14': 'Moratalaz',
    '15': 'Ciudad Lineal',
    '16': 'Hortaleza',
    '17': 'Villaverde',
    '18': 'Villa de Vallecas',
    '19': 'Vicálvaro',
    '20': 'San Blas',
    '21': 'Barajas'
}



report = TableReport(df)

# Guardar reporte HTML
with open('reports/report_demograficos.html', 'w') as f:
    f.write(report.html())
print('\n--> Reporte guardado en reports/report_demograficos.html\n')

# print(f"""Datos de interés:
#       - {df.shape[0]} datos de viviendas.
#       - Precio de venta medio: {round(df['PRICE_SALE'].mean())}€.
#       - Precio de alquiler medio: {round(df['PRICE_RENT'].mean())}€.
#       - Periodo: {min(df[df['YEAR'] == min(df['YEAR'])]['MONTH'])}/{min(df['YEAR'])} - {max(df[df['YEAR'] == max(df['YEAR'])]['MONTH'])}/{max(df['YEAR'])}
# """)