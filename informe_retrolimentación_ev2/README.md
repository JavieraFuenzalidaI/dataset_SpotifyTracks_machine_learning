# Inteligencia musical y predicción de popularidad (Spotify Tracks)

Evaluación Parcial N°2, Machine Learning (MLY1101). Catalina Villegas y Javiera Fuenzalida, sección 002D, docente Ignacio Meza.

Proyecto de Machine Learning sobre 113.549 canciones de Spotify (114 géneros): predice si una canción será **hit** (`popularity >= 70`), cuántos **puntos de popularidad** tendrá y descubre **perfiles sonoros** con PCA y K-Means. Metodología CRISP-DM.

## instalaciones necesarias

pandas, numpy, matplotlib,seaborn, category_encoders, sklearn, imbalanced-learn,lightgbm, xgboost

## Estructura del proyecto

```
├── data/
│   ├── Spotify_Tracks_Dataset.csv          # datos originales
│   └── Spotify_Tracks_Dataset_Clean.csv    # generado por EDA.ipynb
├── notebooks/
│   ├── EDA.ipynb                           # calidad de datos, KPIs y gráficos
│   └── EvParcial2_MachineLearning_VillegasFuenzalida.ipynb   # modelado
├── images/                                 # gráficos generados por los notebooks
├── informe_tecnico.md                      # informe completo
└── README.md
```

La carpeta `models/` no se usa: los modelos se entrenan al ejecutar el notebook (tardan pocos minutos).

## Cómo ejecutar

**Google Colab**

1. Sube al panel de archivos `Spotify_Tracks_Dataset.csv` y abre `EDA.ipynb`. Ejecuta todo: genera `Spotify_Tracks_Dataset_Clean.csv` y los gráficos del EDA.
2. Abre el notebook de modelado, sube `Spotify_Tracks_Dataset_Clean.csv` a la sesión y ejecuta todo. La primera celda instala `category_encoders` e `imbalanced-learn`.

**Local:** respeta la estructura anterior; los notebooks detectan `../data` y `../images` solos.

## Resultados principales

| Tarea | Mejor modelo | Resultado (test) |
|---|---|---|
| Clasificar hit | LightGBM con pesos de clase | ROC-AUC 0,842, PR-AUC 0,312, recall 0,60, precision 0,20 |
| Predecir popularidad | HistGradientBoosting | RMSE 16,5 (línea base 22,2), R² 0,45 |
| Perfiles sonoros | PCA (5 componentes, 81,5 %) + K-Means k = 4 | 4 perfiles; ningún perfil supera 6 % de hits |

Decisiones clave: partición **agrupada por `track_id`** y **estratificada por género** (test con los 114 géneros), codificadores ajustados solo con train y balanceo (SMOTENC) **solo en train**. Detalle completo, registro de decisiones y limitaciones en [`informe_tecnico.md`](informe_tecnico.md).
