# Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)

Proyecto de **Machine Learning (MLY1101, Duoc UC)**, Evaluación Parcial N°3. Caso C: predecir si una canción será *hit* (`popularity >= 70`) a partir de sus atributos de audio y metadatos, comparando un árbol de decisión simple contra dos ensambles (Random Forest y LightGBM), y descubriendo perfiles sonoros con K-Means.

**Integrantes:** Catalina Villegas y Javiera Fuenzalida · **Sección:** 002D · **Docente:** Ignacio Meza

## Resultado principal

| Modelo | PR-AUC en validación cruzada (5 pliegues) | Recall de Hit en test | Precision de Hit en test |
| :---- | :---- | :---- | :---- |
| Árbol de decisión (optimizado) | 0,140 | 0,822 | 0,143 |
| Random Forest (optimizado) | 0,318 | 0,770 | 0,189 |
| **LightGBM (optimizado)** | **0,367** | **0,883** | **0,225** |

* Modelo seleccionado: **LightGBM optimizado** (PR-AUC en test 0,392; lift de 3,96 veces el azar).  
* El modelo funciona como **filtro de candidatos** para revisión humana: detecta el 88 % de los hits, pero solo 1 de cada 4 alertas es un hit real.  
* K-Means encuentra 7 perfiles sonoros, con tasas de hits entre 0 % y 9 %.

El detalle completo está en [`informe_tecnico.md`](http://informe_tecnico.md).

## Estructura del proyecto

```
.
├── data/
│   ├── Spotify_Tracks_Dataset.csv          # datos originales
│   └── Spotify_Tracks_Dataset_Clean.csv    # datos limpios (salida del EDA)
├── notebooks/
│   ├── EDA.ipynb                        # limpieza y análisis exploratorio
│   └── EvParcial3_MachineLearning...       # modelos, optimización, validación y selección
├── images/                                 # gráficos generados por los notebooks
├── informe_tecnico.md
└── README.md
```

Cómo reproducir

### Opción A: Google Colab

1. Sube `Spotify_Tracks_Dataset.csv` a la sesión (ícono de carpeta) y ejecuta `EDA.ipynb`. Genera `Spotify_Tracks_Dataset_Clean.csv`.  
2. Abre `EvParcial3_MachineLearning_VillegasFuenzalida.ipynb`, mantén el CSV limpio en la sesión y ejecuta todas las celdas (**Entorno de ejecución \> Ejecutar todo**).

### Opción B: local

1. Clona el repositorio y crea un entorno virtual:

```sh
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm jupyter
```

2. Abre Jupyter y ejecuta primero `notebooks/EDA.ipynb` y luego `notebooks/EvParcial3_MachineLearning_VillegasFuenzalida.ipynb`. Los notebooks buscan los datos en `../data` automáticamente.

**Tiempo de ejecución aproximado de `EvParcial3_MachineLearning_VillegasFuenzalida.ipynb` en Colab:** la optimización de hiperparámetros demoró 16 s (árbol), 222 s (Random Forest) y 34 s (LightGBM), más la validación cruzada de 5 pliegues de los seis modelos. En total, unos 8 a 10 minutos.

## Contenido de `EvParcial3_MachineLearning_VillegasFuenzalida.ipynb`

1. Preparación de los datos: una fila por canción, variable objetivo, codificación y partición 80/20.  
2. Optimización de hiperparámetros: Grid Search (árbol) y Random Search (Random Forest y LightGBM).  
3. Ensambles y validación cruzada de 5 pliegues: media, desviación estándar y brecha entre train y validación.  
4. Selección del modelo con una regla definida antes de usar el test, más equidad por subgrupos e importancia de variables.  
5. Tabla de cumplimiento de los KPIs.  
6. K-Means para descubrir perfiles sonoros.

## Metodología

El proyecto sigue el estándar **CRISP-DM**: comprensión del negocio, comprensión de los datos, preparación, modelado, evaluación y despliegue (documentación reproducible).

## Datos

* **Fuente:** dataset *Spotify Tracks* (114.000 filas y 114 géneros en el archivo original).  
* **Datos de trabajo:** 37.497 canciones únicas de 49 géneros (los 25 más populares y los 25 menos populares), con 5,68 % de hits.  
* Contiene metadatos públicos de canciones, sin datos personales de oyentes.

## Limitaciones

* La precision es baja (0,225): se requiere revisión humana de las alertas.  
* La selección de géneros usa la popularidad y no refleja el catálogo real.  
* La popularidad refleja lo que Spotify ya promovió (sesgo de retroalimentación).  
* La estabilidad del modelo entre pliegues (0,024) queda levemente sobre la meta de 0,02.