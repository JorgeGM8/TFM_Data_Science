# Análisis del mercado inmobiliario y detección temprana de gentrificación en la ciudad de Madrid

Este proyecto aplica técnicas de ciencia de datos para analizar la evolución del mercado inmobiliario madrileño entre 2011 y 2024, estimar la accesibilidad a la vivienda y detectar patrones de gentrificación a nivel distrital.


## 🗒️ Descripción general

El estudio integra datos de Idealista, el Ayuntamiento de Madrid y el INE, generando más de 140.000 registros.

Se desarrollaron modelos predictivos de precios con LightGBM optimizados con Optuna, junto con el diseño de un Índice de Accesibilidad a la Vivienda (IAV) para medir desigualdades territoriales en compra y alquiler.

El análisis espacial y de clustering permitió identificar zonas en proceso de transformación urbana, demostrando el potencial del aprendizaje automático para la planificación basada en evidencia.

Los resultados se integraron en una interfaz web con Streamlit.


## 👥 Autores
- Samantha Barone Guerra  
- Gaspar Campuzano Gallego  
- Jorge Galeano Maté  
- Andrea Guerra Arguinzones

**Tutor:** Gabriel Valverde Castilla  
**Fecha:** octubre 2025  


## 📁 Estructura del proyecto

```
├── app                    # Interfaz web
│   └── .streamlit
├── data                   # Datos (csv)
│   ├── final
│   ├── processed
│   ├── raw
│   └── webscraping
│       ├── cached_pages
│       └── csv
├── models                 # Modelos y encoders
├── notebooks              # Notebooks
├── reports                # Reportes básicos y gráficas
│   └── figures
├── src                    # Scripts de Python
│   └── webscraping
├── utils                  # Funciones definidas
└── (archivos)
```

## ⚙️ Metodología

1. **Recolección de datos**
   - Web scraping automatizado de Idealista (Python, `requests`, `BeautifulSoup`, `selenium`).
   - Datos oficiales (renta, población, paro, zonas verdes, etc.).

2. **Preprocesamiento**
   - Limpieza, normalización y enriquecimiento de variables.
   - Reconstrucción temporal 2011–2024 mediante proyección proporcional distrital.

3. **Modelado predictivo**
   - Entrenamiento con algoritmos de regresión supervisada: Ridge, KNN, MLP, HistGradientBoosting, LightGBM.
   - Validación temporal con `TimeSeriesSplit` y optimización de hiperparámetros mediante Optuna.

4. **Índice de Accesibilidad (IAV)**
   - Cálculo de IAV de compra y de alquiler, basados en precios (€/m²), superficie mínima y renta media disponible.
   - Representación de desigualdades territoriales en el acceso a la vivienda.

5. **Análisis de gentrificación**
   - Creación de embeddings residenciales y agrupamiento mediante UMAP y KMeans.
   - Identificación de patrones de transformación socioespacial y gradientes de gentrificación.

6. **Interfaz interactiva**
   - Aplicación desarrollada en Streamlit para la visualización, predicción y análisis de los resultados de forma dinámica.


## *️⃣ Tecnologías utilizadas

- Python >=3.12
- Pandas, NumPy, Scikit-learn, PyTorch, LightGBM, Optuna
- Matplotlib, Seaborn, Plotly
- Streamlit
- UMAP, KMeans
- BeautifulSoup, Selenium


## ⚡ Ejecución

Para ejecutar los archivos, se debe crear un entorno virtual. Para ello, hay tres opciones: con UV, con pip y con conda. Los tres son igual de válidos.

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # o venv\Scripts\activate en Windows

# Instalar dependencias con UV (opcional)
uv init
uv sync

# Instalar dependencias con pip
pip install .   # Uso de pyproject.toml (recomendado)
pip install -r requirements.txt   # O uso de archivo de requirements

# Instalar dependencias con conda
conda install --file requirements.txt

# Ejecutar la interfaz web
streamlit run app/app.py
```


## ✅ Licencia y uso
Uso exclusivamente académico.

Los datos obtenidos mediante scraping se procesan de forma anónima y agregada, respetando las políticas del portal Idealista.

---
---

# Real estate market analysis and early detection of gentrification in the city of Madrid

This project applies data science techniques to analyze the evolution of the Madrid real estate market between 2011 and 2024, estimate housing accessibility, and detect gentrification patterns at the district level.


## 🗒️ Overview

The study integrates data from Idealista, the Madrid City Council, and the INE, generating over 140,000 records.

Predictive price models were developed using LightGBM optimized with Optuna, along with the design of a Housing Accessibility Index (HAI) to measure territorial inequalities in buying and renting.

Spatial and clustering analysis allowed the identification of areas undergoing urban transformation, demonstrating the potential of machine learning for evidence-based planning.

The results were integrated into a web interface using Streamlit.


## 👥 Authors

- Samantha Barone Guerra  
- Gaspar Campuzano Gallego  
- Jorge Galeano Maté  
- Andrea Guerra Arguinzones  

**Advisor:** Gabriel Valverde Castilla  
**Date:** October 2025


## 📁 Project Structure

```
├── app                    # Web interface
│   └── .streamlit
├── data                   # Data (csv)
│   ├── final
│   ├── processed
│   ├── raw
│   └── webscraping
│       ├── cached_pages
│       └── csv
├── models                 # Models and encoders
├── notebooks              # Notebooks
├── reports                # Basic reports and plots
│   └── figures
├── src                    # Python scripts
│   └── webscraping
├── utils                  # Defined functions
└── (files)
```


## ⚙️ Methodology

1. **Data collection**
   - Automated web scraping from Idealista (Python, requests, BeautifulSoup, Selenium).
   - Official data (income, population, unemployment, green areas, etc.).
2. **Preprocessing**
   - Cleaning, normalization, and variable enrichment.
   - Temporal reconstruction 2011–2024 through proportional district projection.
3. **Predictive modeling**
   - Training with supervised regression algorithms: Ridge, KNN, MLP, HistGradientBoosting, LightGBM.
   - Temporal validation with TimeSeriesSplit and hyperparameter optimization using Optuna.
4. **Accessibility Index (HAI)**
   - Calculation of purchase and rental HAI based on prices (€/m²), minimum surface, and average available income.
   - Representation of territorial inequalities in housing access.
5. **Gentrification analysis**
   - Creation of residential embeddings and clustering using UMAP and KMeans.
   - Identification of socio-spatial transformation patterns and gentrification gradients.
6. **Interactive interface**
   - Application developed in Streamlit for dynamic visualization, prediction, and analysis of results.


## *️⃣ Technologies used

- Python >=3.12
- Pandas, NumPy, Scikit-learn, PyTorch, LightGBM, Optuna
- Matplotlib, Seaborn, Plotly
- Streamlit
- UMAP, KMeans
- BeautifulSoup, Selenium


## ⚡ Execution

To run the files, a virtual environment must be created. There are three options: with UV, with pip, and with conda. All three are equally valid.

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies with UV (optional)
uv init
uv sync

# Install dependencies with pip
pip install .   # Using pyproject.toml (recommended)
pip install -r requirements.txt   # Or using a requirements file

# Install dependencies with conda
conda install --file requirements.txt

# Run the web interface
streamlit run app/app.py
```


## ✅ License and Use

For academic use only.

Data obtained through scraping is processed anonymously and in aggregate, respecting Idealista’s portal policies.