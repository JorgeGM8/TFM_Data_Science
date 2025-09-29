import pandas as pd
try:
    from utils.calculos import calcular_alquiler_venta, ajustar_precios_simple, ajustar_predicciones_con_limites, ajustar_predicciones_hibrido
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

perifericos = False
try:
    df_idealista_perifericos = pd.read_csv('data/final/inmuebles_unificado_total_final.csv')
    perifericos = True
except FileNotFoundError:
    print('--> Archivo de datos periféricos no encontrado. Se utilizarán solo los de Madrid capital.')
except Exception as e:
    print(f'--> Error inesperado con datos periféricos: {e}\nSe utilizarán solo los de Madrid capital.')

# --- UNIÓN DE CSV ---
df_completo = df_gentrificacion.merge(df_idealista, how='left', on=['Distrito'])

if perifericos:
    df_completo_perifericos = df_gentrificacion.merge(df_idealista_perifericos, how='left', on=['Distrito'])

print('--> Unidos csv de gentrificacion y de inmuebles de 2025.')

# --- APLICACIÓN DE PREDICCIÓN A PASADO (REGLA DE 3) ---
# Predicción para alquileres
df_completo = calcular_alquiler_venta(df_completo, 'alquiler')

if perifericos:
    df_completo_perifericos = calcular_alquiler_venta(df_completo_perifericos, 'alquiler')

print('--> Predicciones a pasado de alquiler completadas.')

# Predicción para ventas
df_completo = calcular_alquiler_venta(df_completo, 'venta')

if perifericos:
    df_completo_perifericos = calcular_alquiler_venta(df_completo_perifericos, 'venta')

print('--> Predicciones a pasado de venta completadas.')

# Ajustar predicciones
df_ajustado = ajustar_predicciones_hibrido(df_completo, alpha=0.8, max_desv_venta=0.2, max_desv_alquiler=0.1, seed=42)

if perifericos:
    df_ajustado_perifericos = ajustar_predicciones_hibrido(df_completo_perifericos,
                                                           alpha=0.8,
                                                           max_desv_venta=0.2,
                                                           max_desv_alquiler=0.1,
                                                           seed=42)

# Guardado intermedio para mantener columnas que eliminaremos
df_ajustado.to_csv('data/processed/viviendas_2011_2024_nodrop.csv', index=False)

if perifericos:
    df_ajustado_perifericos.to_csv('data/processed/viviendas_perifericos_2011_2024_nodrop.csv', index=False)

# Eliminación de fila de "Precio" (valores obsoletos) y otras columnas que ya no se usarán
columnas_eliminar = ['Precio', 'Precio_venta', 'Precio_alquiler', 'Precio_real']
df_ajustado = df_ajustado.drop(columns=columnas_eliminar)

if perifericos:
    df_ajustado_perifericos = df_ajustado_perifericos.drop(columns=columnas_eliminar)

print(f'--> Eliminadas columnas: {columnas_eliminar}.')

# --- GUARDADO DE CSV FINAL ---
df_ajustado.to_csv('data/final/viviendas_2011_2024.csv', index=False)
print('--> Datos finales de Madrid capital guardados en "data/final/viviendas_2011_2024.csv".')

if perifericos:
    df_ajustado_perifericos.to_csv('data/final/viviendas_perifericos_2011_2024.csv', index=False)
    print('--> Datos finales de periféricos guardados en "data/final/viviendas_perifericos_2011_2024.csv".')