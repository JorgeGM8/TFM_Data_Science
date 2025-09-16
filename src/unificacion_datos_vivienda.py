
# import pandas as pd

# # Rutas de los ficheros CSV
# PATH_GENERAL = "data/raw/inmuebles_+_variables_distrito_anio.csv"
# PATH_VARIABLES = "data/processed/gentrificacion_madrid.csv"
# ANIO_INMUEBLES = 2025

# # Cargar datos
# df_gen = pd.read_csv(PATH_GENERAL)
# df_var = pd.read_csv(PATH_VARIABLES)

# # ------------------------------------- NORMALIZACIÓN, AÑO Y MERGE DE DATAFRAMES ------------------------------------- #

# # Normalizar nombres de distrito directamente en ambos dataframes
# df_gen["distrito_std"] = (df_gen["distrito"].str.lower().str.normalize('NFKD')
#                           .str.encode('ascii', errors='ignore').str.decode('ascii').str.replace('[ -]', '', regex=True))
# df_var["distrito_std"] = (df_var["Distrito"].str.lower().str.normalize('NFKD')
#                           .str.encode('ascii', errors='ignore').str.decode('ascii').str.replace('[ -]', '', regex=True))

# # Añadir año a df_gen y asegurar tipo correcto en df_var
# df_gen["anio"] = ANIO_INMUEBLES
# df_var["Ano"] = pd.to_numeric(df_var["Ano"], errors="coerce").astype("Int64")

# # Merge directo
# df_final = df_gen.merge(df_var, left_on=["distrito_std", "anio"], right_on=["distrito_std", "Ano"], how="left", suffixes=("", "_soc"))

# # Información del resultado
# print(f"Unificación completada: {len(df_final)} filas, {len(df_final.columns)} columnas")
# print(df_final[["distrito", "anio", "precio_num", "Precio_venta", "Precio_alquiler"]].head())

# # ---------------------------------------------- VOLCADO A CSV Y PARQUET --------------------------------------------- #

# # Definir directorio de salida y nombre base de los archivos
# output_dir = PATH_GENERAL
# base_name = "inmuebles_+_variables_distrito_anio"

# # Guardar el DataFrame en CSV y Pickle (texto legible y binario rápido)
# df_final.to_csv(f"{base_name}.csv", index=False, encoding="utf-8")
# df_final.to_pickle(f"{base_name}.pkl")

# # Intentar guardar también en Parquet (formato columnar más eficiente);
# try:
#     df_final.to_parquet(f"{base_name}.parquet", index=False)
# except ImportError:
#     print("Parquet no disponible - instala pyarrow o fastparquet")

# print(f"Dataset guardado: {len(df_final)} filas, {len(df_final.columns)} columnas")

# # ------------------------------------------------- ANALISIS DE DATOS ------------------------------------------------ #

# df = df_final

# # Estructura general
# print(f"Dimensiones del dataset: {df.shape}")
# print(f"\nTipos de datos principales:")
# print(df.dtypes.value_counts())

# # Comparación entre 'anio' y 'Ano'
# print(f"\nValores únicos en 'anio': {sorted(df['anio'].dropna().unique())}")
# print(f"Valores únicos en 'Ano':  {sorted(df['Ano'].dropna().unique())}")

# df_diff = df[df["anio"] != df["Ano"]][["distrito", "anio", "Ano", "precio_num"]]
# print(f"\nFilas donde 'anio' ≠ 'Ano': {len(df_diff)}")
# if len(df_diff) > 0:
#     print(df_diff.head(10))

# # Valores nulos
# print(f"\nPorcentaje de valores nulos por columna:")
# null_pct = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
# print(null_pct[null_pct > 0].head(15))

# # Estadísticas numéricas clave
# print(f"\nEstadísticas descriptivas - variables precio:")
# price_cols = ['precio_num', 'Precio_venta', 'Precio_alquiler']
# existing_cols = [col for col in price_cols if col in df.columns]
# if existing_cols:
#     print(df[existing_cols].describe().round(0))

# # Variables categóricas clave
# print(f"\nDistribución de operaciones:")
# print(df["operacion"].value_counts(dropna=False))

# print(f"\nTop 10 distritos:")
# print(df["distrito"].value_counts(dropna=False).head(10))
