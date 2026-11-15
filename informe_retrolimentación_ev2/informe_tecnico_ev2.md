# Informe técnico: Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)

| Campo | Detalle |
|---|---|
| Evaluación | Evaluación Parcial N°2, Machine Learning (MLY1101) |
| Integrantes | Catalina Villegas y Javiera Fuenzalida |
| Sección | 002D |
| Docente | Ignacio Meza |
| Caso | C: Inteligencia musical y predicción de popularidad de canciones |

---

## 1. Descripción del problema de negocio

Las plataformas de streaming reciben miles de canciones nuevas cada día. Para discográficas, creadores de contenido y la propia plataforma, anticipar la popularidad de una pista permite priorizar inversión en marketing, armar listas editoriales y aumentar el engagement.

El problema es **identificar qué atributos acústicos y metadatos se asocian con la popularidad de una canción y construir modelos que la anticipen**, de modo que el equipo de negocio revise primero las canciones con más probabilidad de éxito.

Se abordan dos preguntas:

1. **Clasificación:** ¿la canción será un *hit* (`popularity >= 70`)?
2. **Regresión:** ¿cuántos puntos de popularidad (0 a 100) tendrá?

Además, se usa aprendizaje no supervisado para descubrir **perfiles sonoros** de canciones.

## 2. Objetivos del proyecto

**Objetivo general:** desarrollar una solución de Machine Learning para predecir la popularidad de canciones en Spotify a partir de sus atributos acústicos y metadatos.

**Objetivos específicos:**

- Aplicar el proceso completo de ciencia de datos bajo CRISP-DM.
- Preparar los datos evitando fuga de información entre entrenamiento y prueba.
- Comparar dos modelos supervisados de clasificación (hit / no hit) y modelos de regresión (popularidad).
- Descubrir perfiles sonoros mediante PCA y K-Means e interpretarlos para el negocio.
- Interpretar las métricas y traducirlas en recomendaciones.

## 3. KPIs

| KPI | Definición | Meta | Valor obtenido | ¿Cumple con la meta? |
|---|---|---|---|---|
| Tasa global de hits | % del catálogo con `popularity >= 70` |META| **4,82 %** | - |
| Popularidad promedio por género | Géneros con mayor atractivo comercial |META| Mayor *hit ratio*: pop 31,9 %, dance 25,3 %, electro 24,1 %, k-pop 22,5 %, house 22,1 % | - |
| Índice de afinidad sonora | Correlación entre atributos de audio y popularidad |META| Débil: todas las correlaciones con `popularity` son menores a 0,10 | - |
| Duración óptima | Duración con mayor popularidad promedio |META| **3 a 5 minutos** (cerca de 34,8 puntos) | - |
| Recall de Hit | Hits reales que el modelo detecta |META| 0,14 a 0,67 según modelo | - |
| Precision de Hit | Alertas de hit que realmente lo son |META| 0,15 a 0,49 según modelo | - |
| RMSE y MAE de popularidad | Error promedio en puntos de popularidad |META| Mejor modelo: RMSE 16,5 y MAE 11,3 (línea base: 22,2 y 18,8) | - |

## 4. Fuentes de datos

- **Dataset:** `Spotify_Tracks_Dataset.csv`: 114.000 canciones, 20 variables más un índice, 114 géneros (aprox. 1.000 canciones por género).
- **Variables:** metadatos (`track_id`, `artists`, `album_name`, `track_name`, `explicit`, `track_genre`, `duration_ms`), atributos de audio (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `key`, `mode`, `time_signature`) y la variable `popularity` (0 a 100).
- **Dataset limpio:** `EDA.ipynb` genera `Spotify_Tracks_Dataset_Clean.csv` (113.549 filas, 20 columnas), insumo del notebook de modelado.
- **Herramientas:** Python en Google Colab; pandas, NumPy, matplotlib, seaborn, scikit-learn, LightGBM, XGBoost, imbalanced-learn y category_encoders.

## 5. Preparación y análisis exploratorio (EDA)

### 5.1 Calidad de datos

| Hallazgo | Cantidad | Tratamiento |
|---|---|---|
| Filas duplicadas exactas | 450 | Eliminadas |
| Fila con `artists`, `album_name` y `track_name` nulos | 1 | Eliminada (también tenía duración 0) |
| `tempo = 0` | 157 (0,138 %) | Se reemplaza por nulo y se imputa con la mediana |
| `time_signature = 0` | 163 (0,144 %) | Se reemplaza por nulo y se imputa con la mediana |
| Valores fuera de rango en el resto de variables | 0 | Sin acción |
| `track_id` repetido | 23.809 filas | Una misma canción aparece en varios géneros; se usa para agrupar la partición |
| `popularity = 0` | 15.843 (14,0 %) | Se mantienen (ver decisión 3) |

Resultado: 114.000 → 113.549 filas.

### 5.2 Patrones encontrados

- **Desbalance de la clase Hit:** solo 4,82 % de las canciones son hit (ver `images/graficos_HIT.png`).
- **El género importa:** el hit ratio va de 31,9 % (pop) a 0 % en varios géneros; 23 géneros no tienen ningún hit (ver `images/graficos_generos.png`).
- **Atributos de audio con relación débil:** `instrumentalness` (-0,10), `loudness` (0,05), `speechiness` (-0,05), `valence` (-0,04), `danceability` (0,03), `acousticness` (-0,02) (ver `images/mapa_calor.png` y `images/heatmap_correlacion_is_hit.png`).
- **Duración:** la popularidad promedio es 28,6 bajo 2 minutos, cerca de 34,8 entre 3 y 5, 29,3 entre 6 y 10 y 24,7 sobre 10 (ver `images/grafico_relacion.png`).
- **Artistas y tempo:** se compara la popularidad de los 15 artistas con más canciones (`images/boxplot_artistas_popularidad.png`) y la distribución del tempo en las canciones hit (`images/histograma_tempo_hits.png`).

### 5.3 Transformaciones

- `duration_min` = `duration_ms` / 60.000.
- `is_hit` = 1 si `popularity >= 70`.
- `log1p` en `speechiness`, `acousticness`, `instrumentalness`, `liveness` y `duration_min` (muy asimétricas). No aprende nada de los datos, por lo que se aplica antes de dividir.
- Se eliminan `album_name`, `track_name` (texto libre de alta cardinalidad) y `duration_ms` (duplica a `duration_min`). `track_id` se conserva solo para agrupar la partición.

### 5.4 Partición y codificación (sin fuga de información)

- **Partición 80/20** con `StratifiedGroupKFold`, **agrupada por `track_id`** y **estratificada por género**. Resultado: 90.786 filas de train y 22.763 de test; 0 % de las canciones de test aparecen en train; el test contiene los **114 géneros**.
- **Test sin modificar:** no se balancea ni contiene datos sintéticos.
- **Ajuste solo con train:** género con `BinaryEncoder` (7 columnas); artista con `TargetEncoder` de scikit-learn (suavizado 10, validación cruzada interna); continuas con `StandardScaler`.
- **Balanceo solo en train:** `SMOTENC` (86.362 hits y 86.362 no hits tras el sobremuestreo).

## 6. Metodología (CRISP-DM)

| Fase | Qué se hizo |
|---|---|
| 1. Comprensión del negocio | Problema, objetivos y KPIs (secciones 1 a 3) |
| 2. Comprensión de los datos | Calidad, patrones y variables relevantes en `EDA.ipynb` (sección 5.1 y 5.2) |
| 3. Preparación de los datos | Limpieza, transformaciones, partición agrupada y codificación sin fuga (secciones 5.3 y 5.4) |
| 4. Modelado | Dos clasificadores, cuatro regresores y PCA con K-Means (sección 7) |
| 5. Evaluación | Matriz de confusión, precision, recall, F1, ROC-AUC y PR-AUC; RMSE, MAE y R²; silueta (sección 8) |
| 6. Despliegue | Notebooks reproducibles, este informe y el `README.md` |

## 7. Modelado

### 7.1 Clasificación (¿es hit?)

Con 95,4 % de canciones "No Hit" en test, un modelo que siempre responda "No Hit" tendría 95,4 % de exactitud, por lo que la exactitud no sirve. La métrica principal es la **matriz de confusión**, junto con precision, recall y F1 de la clase Hit, ROC-AUC y PR-AUC.

| Algoritmo | Justificación |
|---|---|
| RandomForest | Robusto, pocos hiperparámetros, buena referencia |
| LightGBM | Gradient boosting eficiente en datos tabulares; admite ponderar la clase minoritaria (`scale_pos_weight` = 19,5) |

Cada uno se entrena sin sobremuestreo (LightGBM con pesos de clase) y con SMOTENC aplicado solo a train. Matrices en `images/matriz_RFC.png`, `images/matriz1_LGBMC.png`, `images/matriz2_RFC.png` e `images/matriz2_LGBMC.png`.

### 7.2 Regresión (¿cuánta popularidad?)

Se compara contra una **línea base** que predice siempre el promedio de train. Modelos: Ridge (referencia lineal e interpretable), RandomForest (no linealidades), HistGradientBoosting y XGBoost (boosting). Gráficos en `images/regresion.png` e `images/regresion_real_vs_predicho.png`.

### 7.3 No supervisado: PCA y K-Means

- **Datos:** solo los 9 atributos de audio (sin `popularity`, `is_hit`, artista ni género), con una fila por canción única (89.740 canciones), para que los grupos reflejen el sonido y no la variable objetivo.
- **PCA:** 5 componentes explican 81,5 % de la varianza (umbral 80 %) (`images/pca_varianza.png`).
- **K-Means:** número de grupos con codo y silueta (`images/kmeans_k.png`). La silueta es máxima con k = 2 (0,292), pero solo separa suave de intenso; entre k = 3 y k = 8 va de 0,206 a 0,229 y el codo no muestra quiebre claro. Se elige **k = 4** por interpretabilidad. La silueta moderada indica grupos que se traslapan, esperable en un continuo musical.

## 8. Resultados e interpretación

### 8.1 Clasificación (test: 22.763 canciones, 1.044 hits reales = 4,59 %)

| Modelo | Precision Hit | Recall Hit | F1 Hit | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RandomForest sin balanceo | 0,495 | 0,142 | 0,220 | 0,812 | 0,268 |
| LightGBM con pesos de clase | 0,196 | 0,604 | 0,296 | **0,842** | **0,312** |
| RandomForest + SMOTENC | 0,294 | 0,374 | **0,329** | 0,826 | 0,251 |
| LightGBM + SMOTENC | 0,151 | **0,674** | 0,247 | 0,815 | 0,273 |

Matriz de confusión en conteos (sobre el test):

| Modelo | Hits detectados (VP) | Hits perdidos (FN) | Falsas alarmas (FP) |
|---|---|---|---|
| RandomForest sin balanceo | 148 | 896 | 151 |
| LightGBM con pesos de clase | 631 | 413 | 2.589 |
| RandomForest + SMOTENC | 390 | 654 | 935 |
| LightGBM + SMOTENC | 704 | 340 | 3.955 |

**Lectura:**

- **Sin balanceo** el modelo casi no detecta hits: encuentra 148 de 1.044 (14 %). Es preciso, pero pierde el 86 % de los éxitos.
- **Ponderar o sobremuestrear sube el recall a costa de precision**: LightGBM + SMOTENC detecta 704 hits (67 %), pero genera 3.955 falsas alarmas.
- **Lift sobre el azar:** un hit aparece en el 4,6 % de las canciones. Con precision 0,196, LightGBM con pesos de clase concentra hits 4,3 veces más que elegir al azar; RandomForest + SMOTENC, 6,4 veces, aunque con menos cobertura.
- **ROC-AUC y PR-AUC:** la mejor discriminación global es LightGBM con pesos de clase (0,842 y 0,312; el azar da 0,05 en PR-AUC).
- **Modelo recomendado como equilibrio:** LightGBM con pesos de clase. Si perder un hit es mucho más costoso que revisar de más, conviene LightGBM + SMOTENC; si la revisión es costosa, RandomForest + SMOTENC.

Variables más importantes (LightGBM + SMOTENC, `images/importancia_variables.png`): `artists_encoded` más que duplica a la segunda variable. El género no aparece entre las diez principales, aunque su efecto puede estar repartido en sus siete columnas binarias.

### 8.2 Regresión

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| Base (promedio) | 22,220 | 18,800 | 0,000 |
| Ridge | 17,720 | 13,033 | 0,364 |
| RandomForest | 16,543 | 11,208 | 0,446 |
| **HistGradientBoosting** | **16,464** | 11,253 | **0,451** |
| XGBoost | 16,591 | 11,386 | 0,443 |

El mejor modelo (HistGradientBoosting) reduce el RMSE en 26 % y el MAE en 40 % respecto de la línea base, y explica 45 % de la variación. Los tres modelos de árboles quedan casi empatados, por lo que la elección entre ellos no es crítica. Un error típico de unos 11 puntos sigue siendo alto: sirve para ordenar canciones, no para predecir su popularidad exacta.

### 8.3 Perfiles sonoros (K-Means, k = 4)

| Grupo | Canciones | Perfil sonoro | Géneros típicos | Tasa de hits | Popularidad media |
|---|---|---|---|---|---|
| 0 | 18.667 | Suave y acústico: energía 0,27, volumen -15,3 dB | sleep, ambient, new-age | 1,6 % | 30,4 |
| 1 | 36.296 | Bailable y positivo: danceability 0,68, valence 0,65 | salsa, forró, kids | 5,2 % | 34,6 |
| 2 | 26.523 | Intenso: energía 0,82, sin componente acústico, tempo 137 | black-metal, grindcore, death-metal | 2,7 % | 33,0 |
| 3 | 8.254 | Hablado o en vivo: speechiness y liveness altos | comedy, pagode, sertanejo | 2,3 % | 33,9 |

La base de comparación es 3,5 % de hits en las canciones únicas (4,8 % si se cuentan las filas repetidas por género). El grupo 1 triplica la tasa del grupo 0, pero **ningún perfil supera el 6 %**: el sonido por sí solo no determina que una canción sea hit.

### 8.4 Recomendaciones para el negocio

1. **Usar el clasificador como filtro de candidatos**, no como decisión automática. Revisar las canciones marcadas permite encontrar hits con una frecuencia 4 a 6 veces mayor que al azar, pero entre 1 de 2 y 1 de 7 alertas son correctas según el modelo.
2. **Elegir el modelo según el costo del error:** recall si perder un hit es más costoso; precision si la revisión es costosa.
3. **Segmentar playlists y campañas por perfil sonoro** (por ejemplo, el grupo 1 para listas bailables), sin esperar que el perfil prediga el éxito.
4. **Dar peso al artista y al género** en las decisiones de promoción: pesan más que cualquier atributo de audio.
5. **Mejorar el modelo con variables ausentes**: fecha de lanzamiento, inversión en promoción y presencia en listas.

## 9. Registro de decisiones

| N° | Decisión | Motivo y dato |
|---|---|---|
| 1 | Se eliminan `track_id`, `album_name`, `track_name` y `duration_ms` como variables | Identificador y texto libre sin valor predictivo; `duration_ms` duplica a `duration_min` |
| 2 | Hit si `popularity >= 70` | Solo 4,8 % de las canciones son hit: clases muy desbalanceadas |
| 3 | Se mantienen las canciones con `popularity = 0` (14,0 %) | No se pueden distinguir canciones sin datos de canciones realmente impopulares; se registra como limitación |
| 4 | Partición agrupada por `track_id` | 23.809 filas repiten `track_id`; con partición aleatoria, 31 % de las canciones de test también estaba en train |
| 5 | Partición estratificada por género | El test contiene los 114 géneros |
| 6 | Codificadores y escalador se ajustan solo con train | Evita que el codificador de artista vea la variable objetivo del test |
| 7 | Se balancea solo train (SMOTENC) | El test queda con datos reales; datos sintéticos en test invalidarían las métricas |
| 8 | `log1p` en cinco variables asimétricas y `StandardScaler` en las continuas | `valence` no se escala porque ya está entre 0 y 1 |
| 9 | Métrica principal: matriz de confusión, recall y PR-AUC de Hit | Predecir siempre No Hit da 95 % de exactitud |
| 10 | PCA (80 % de varianza) y K-Means solo con atributos de audio, k = 4 | Perfiles sonoros sin sesgo de la variable objetivo; k = 4 por interpretabilidad |
| 11 | Limitación: 23 géneros no tienen ningún hit | El modelo no puede aprender hits de esos géneros |
| 12 | Limitación: el codificador de artista no existe para un artista nuevo | 24 % de las canciones de test son de artistas no vistos en train; reciben el promedio global |
| 13 | Limitación: 1.000 canciones por género | Distribución artificial que no refleja el catálogo real de Spotify |
| 14 | Se corrige el orden de etiquetas de las matrices de confusión | En la versión anterior, `Es Hit` y `No es Hit` estaban invertidas respecto de la matriz; se conservan nombres de archivo, títulos y ejes |
| 15 | Resultados más bajos que en la versión anterior | Se eliminó la fuga de información; las métricas anteriores estaban infladas (ver sección 10) |

## 10. Comparación con la versión anterior y fuga de información

La primera versión del modelado usaba una partición aleatoria y ajustaba el codificador de artista con todos los datos. Eso inflaba los resultados por dos razones: una misma canción quedaba en train y en test (31 % del test), y el codificador de artista veía la popularidad de las canciones de test.

| Métrica | Versión anterior | Versión actual (sin fuga) |
|---|---|---|
| R² RandomForest (regresión) | 0,70 | 0,45 |
| RMSE RandomForest | 12,2 | 16,5 |
| Recall Hit, RandomForest sin balanceo | 0,60 | 0,14 |
| Recall Hit, LightGBM con pesos de clase | 0,98 | 0,60 |

Las métricas actuales son más bajas, pero **son las que reflejan el desempeño con canciones realmente nuevas**.

## 11. Ética, sesgos y limitaciones

- **Sesgo de selección:** hay 1.000 canciones por género, una distribución artificial que no refleja el catálogo real.
- **Sesgo de retroalimentación:** la popularidad refleja lo que Spotify ya promovió; un modelo entrenado con ella reproduce esas decisiones y favorece a artistas ya establecidos.
- **Datos faltantes:** 14 % de las canciones tiene `popularity = 0`, que mezcla "sin datos" con "impopular".
- **Privacidad:** el dataset contiene metadatos públicos de canciones y no incluye datos personales de oyentes.
- **Uso responsable:** las predicciones son una herramienta de apoyo; no deben usarse para descartar obras de valor artístico por tener métricas alejadas del estándar comercial.
- **Limitaciones técnicas:** 24 % de las canciones de test son de artistas no vistos; el modelo no puede aprender hits de los 23 géneros sin ninguno; el perfil sonoro explica poco el éxito.

## 12. Reproducibilidad

Orden de ejecución: `notebooks/EDA.ipynb` (genera el dataset limpio y los gráficos del EDA) y luego `notebooks/EvParcial2_MachineLearning_VillegasFuenzalida.ipynb` (modelado). Ambos detectan sus rutas automáticamente en local y en Google Colab. Instrucciones en `README.md`.
