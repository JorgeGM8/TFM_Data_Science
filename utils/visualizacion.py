import matplotlib.pyplot as plt
import pandas as pd

def precios_distrito_ano(df: pd.DataFrame, operacion: str, valor: str):
    """
    Gráfica que muestra valores anuales y por distrito del precio medio de venta o alquiler, real o predicho.

    Parameters
    ----------
    df : Dataframe de Pandas
        Dataframe de donde se sacan los datos.
    operacion : str
        Seleccionar "venta" o "alquiler".
    valor : str
        Seleccionar "real" o "predicho".
    """
    
    operacion = operacion.lower()
    valor = valor.lower()

    plt.figure(figsize=(12, 6))

    # Calcular precio medio por distrito y año para ventas o alquileres
    if valor == 'real':
        mean_prices = df[df['Operacion'] == operacion].groupby(['Ano', 'Distrito'], as_index=False).agg(
            Precio_total=pd.NamedAgg(column=f'Precio_{operacion}', aggfunc=lambda x: (x * df.loc[x.index, 'Tamano']).mean())
        )
    elif valor == 'predicho':
        mean_prices = df[df['Operacion'] == operacion].groupby(['Ano', 'Distrito'])['Precio_predicho'].mean().reset_index()

    distritos = df['Distrito'].unique()

    for year in mean_prices['Ano'].unique():
        year_data = mean_prices[mean_prices['Ano'] == year]
        # Ordenar por distrito
        year_data = year_data.set_index('Distrito').reindex(distritos).reset_index()
        if valor == 'real':
            plt.plot(year_data['Distrito'], year_data['Precio_total'], label=str(year))
        else:
            plt.plot(year_data['Distrito'], year_data['Precio_predicho'], label=str(year))

    plt.xticks(rotation=45, ha='right')
    plt.grid()
    plt.xlabel('Distrito')
    plt.ylabel(f'Precio de {operacion} medio (€)')
    plt.title(f'Precio de {operacion} medio {valor} por distrito y año')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'../reports/figures/precio_{operacion}_{valor}.png', dpi=300)
    print(f'--> Gráfica guardada en reports/figures/precio_{operacion}_{valor}.png')
    plt.show()

def comparar_precios(df: pd.DataFrame, operacion: str, ajustado: bool):
    """
    Gráfica que muestra comparativa de precio medio real vs predicho, en general, para cada distrito. Para venta o alquiler.

    Parameters
    ----------
    df : Dataframe de Pandas
        Dataframe de donde se sacan los datos.
    operacion : str
        Seleccionar "venta" o "alquiler".
    tipo : str
        "True" para indicar que el precio predicho ha sido ajustado; "False" para indicar que aún no.
    """
    if ajustado:
        col_predicho = 'Precio_ajustado'
        momento = 'posterior'
    else:
        col_predicho = 'Precio_predicho'
        momento = 'previo'
    distritos = df['Distrito'].unique()

    plt.figure(figsize=(12, 6))

    # Reindexar ambos para mantener orden correcto de distritos
    mean_real = df[df['Operacion'] == operacion].groupby('Distrito').agg(
        precio_completo=(f'Precio_{operacion}', lambda x: (x * df.loc[x.index, 'Tamano']).mean())
    ).reindex(distritos)
    mean_predicted = df[df['Operacion'] == operacion].groupby('Distrito')[col_predicho].mean().reindex(distritos)

    # Plot de ambos
    plt.plot(mean_real.index, mean_real['precio_completo'], 'b-', label='Precio real')
    plt.plot(mean_predicted.index, mean_predicted.values, 'r-', label='Precio predicho')

    plt.xticks(rotation=45, ha='right')
    plt.grid()
    plt.xlabel('Distrito')
    plt.ylabel('Precio (€)')
    plt.title(f'Comparación de precios de {operacion} reales vs predichos por distrito (periodo 2011-2024) - {momento} a ajuste')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../reports/figures/precio_{operacion}_real_vs_predicho_{momento}.png', dpi=300)
    print(f'--> Gráfica guardada en reports/figures/precio_{operacion}_real_vs_predicho_{momento}.png')
    plt.show()