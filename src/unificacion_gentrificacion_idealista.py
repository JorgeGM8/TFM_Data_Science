import pandas as pd
try:
    from utils.calculos import calcular_alquiler_venta, ajustar_predicciones
except ModuleNotFoundError:
    print('--> Usa "python -m src.unificacion_gentrificacion_idealista" para importar correctamente las funciones.')
    quit()
except ImportError as e:
    print(f'--> Error de importación: {e}')
    quit()
except Exception as e:
    print(f'--> Error inesperado: {type(e).__name__}: {e}')
    quit()

# --- LECTURA DE DATASETS ---
try:
    df_gentrificacion = pd.read_csv('data/final/gentrificacion_madrid.csv')
    df_idealista = pd.read_csv('data/final/inmuebles_unificado_total_final.csv')
except FileNotFoundError:
    print('--> Archivos no encontrados, comprueba la ruta y los nombres.')
    quit()
except Exception as e:
    print(f'--> Error inesperado: {e}')
    quit()

# --- UNIÓN DE CSV ---
df_completo = df_gentrificacion.merge(df_idealista, how='left', on=['Distrito'])

print('--> Unidos csv de gentrificacion y de inmuebles de 2025.')

# --- APLICACIÓN DE PREDICCIÓN A PASADO (REGLA DE 3) ---
# Predicción para alquileres
df_completo = calcular_alquiler_venta(df_completo, 'alquiler')
print('--> Predicciones a pasado de alquiler completadas.')

# Predicción para ventas
df_completo = calcular_alquiler_venta(df_completo, 'venta')
print('--> Predicciones a pasado de venta completadas.')

# Guardado intermedio para mantener columnas que eliminaremos
df_completo.to_csv('data/processed/viviendas_2011_2024_nodrop.csv', index=False)

# Ajustar predicciones
df_ajustado = ajustar_predicciones(df_completo)
df_ajustado.to_csv('data/final/prueba.csv', index=False)

# Eliminación de fila de "Precio" (valores obsoletos) y otras columnas que ya no se usarán
df_completo = df_completo.drop(columns=['Precio', 'Precio_venta', 'Precio_alquiler'])
print('--> Eliminadas columnas de "Precio", "Precio_venta" y "Precio_alquiler".')

# --- GUARDADO DE CSV FINAL ---
df_completo.to_csv('data/final/viviendas_2011_2024.csv', index=False)
print('--> Datos finales guardados en "data/final/viviendas_2011_2024.csv".')