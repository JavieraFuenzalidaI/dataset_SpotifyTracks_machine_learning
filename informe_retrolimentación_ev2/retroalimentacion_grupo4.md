# Retroalimentación EP2 – Machine Learning (MLY1101)

**Grupo 4 – Caso C: Spotify Tracks**
**Integrantes:** Javiera Fuenzalida, Catalina Villegas
**Repositorio:** `JavieraFuenzalidaI/ev2_machine_learning`

---

## Resultado de la entrega (grupal)

| Indicador | Ponderación | Nivel de logro |
|---|---|---|
| IE5: Aplicación de la metodología para el desarrollo del modelo | 20% | Buen desempeño (80%) |
| IE6: Construcción de modelos de aprendizaje supervisado | 30% | Muy buen desempeño (100%) |
| IE7: Implementación de modelos de aprendizaje no supervisado | 30% | Muy buen desempeño (100%) |
| IE8: Interpretación del desempeño del modelo | 20% | Muy buen desempeño (100%) |

El avance respecto de la EP1 es muy grande. El informe técnico es claro y completo, y el **registro de decisiones** es una muy buena práctica: cada decisión tiene su motivo y su dato. Lo más destacable es la sección 10: detectaron que su primera versión tenía fuga de datos, la corrigieron y mostraron con honestidad que las métricas bajaron (R² 0,70 → 0,45). Eso es exactamente lo que se espera de un proyecto serio.

La nota individual también considera la **defensa oral**, donde no pudieron explicar parte de lo que está en su propio código (ver la última sección).

---

## Lo que hicieron bien 👍

- **Partición sin fuga:** `StratifiedGroupKFold` agrupado por `track_id` y estratificado por género, con verificación (0% de canciones de test en train y 114 géneros en test).
- **Codificadores ajustados solo con train** y balanceo solo en train. El test se mantiene con datos reales.
- **Comparación con y sin balanceo** para dos algoritmos, con las matrices de confusión en conteos (VP, FN, FP) y no solo en porcentajes.
- **Lift sobre el azar** y recomendación de modelo según el costo del error, que es una lectura de negocio muy bien hecha.
- **Clustering sobre canciones únicas** y solo con audio, con perfiles, géneros típicos y tasa de hits por grupo.
- **Limitaciones medidas:** el 24% de los artistas de test no se vieron en train, y 23 géneros no tienen ningún hit.

---

## Qué mejorar y cómo

### IE5 – Los KPIs necesitan meta

La tabla de KPIs solo trae el "valor obtenido". Sin meta no se puede saber si el proyecto cumplió. Por ejemplo:

| KPI | Meta | Resultado | ¿Cumple? |
|---|---|---|---|
| Recall de Hit | ≥ 0,60 | 0,604 (LightGBM con pesos) | Sí |
| Lift de precisión sobre el azar | ≥ 3× | 4,3× | Sí |
| Mejora de MAE vs. línea base | ≥ 20% | 40% | Sí |

Las metas se fijan **antes** de entrenar y no se cambian después.

### Para seguir mejorando (no afecta esta nota)

- **Línea base de regresión:** comparan contra la media global. Con el género disponible, la línea base exigente es la **media del género**. Así se mide cuánto aporta el audio por encima del género.
- **Ridge con `alpha` por defecto:** justifiquen el valor (en la EP3 lo ajustaremos con validación cruzada).

---

## Sobre la defensa oral

**La regla es simple: si está en la presentación o en el código, tienen que poder explicarlo.**

**1. ¿Qué significa la columna `key`?** Es la **tonalidad** estimada de la canción, un entero de 0 a 11 (0 = Do, 1 = Do♯/Re♭, …, 11 = Si; −1 si no se detecta). Es una variable **nominal y cíclica**: el 11 y el 0 están "al lado", así que no se puede tratar como un número. Ustedes la marcaron como categórica en SMOTENC, lo que estuvo bien, pero tienen que poder explicar por qué.

**2. ¿Qué parámetros tiene Ridge? ¿Qué es `solver`?**
- `alpha`: fuerza de la penalización L2. Más alto significa coeficientes más pequeños y menos sobreajuste. Usaron 1,0, el valor por defecto.
- `fit_intercept`, `max_iter` y `tol`.
- `solver`: el **algoritmo numérico** con que se resuelve la optimización (`auto`, `cholesky`, `lsqr`, `sag`/`saga` para datos grandes). No cambia el modelo, solo la forma de calcularlo.

**3. SMOTE con variables categóricas.** Su código usa **SMOTENC** y le indica qué columnas son categóricas. Lo hicieron bien, pero en la defensa no lo supieron explicar. Lo que había que decir:
- SMOTE crea canciones sintéticas **interpolando** entre vecinos. En una columna 0/1 eso produce valores como 0,37, que no existen.
- **SMOTENC** resuelve eso: a las columnas categóricas les asigna el valor más frecuente entre los vecinos, en vez de interpolar.
- **Una falla que no vieron:** con `BinaryEncoder`, el género queda repartido en 7 bits, y SMOTENC trata cada bit como una categórica **independiente**. Puede armar una combinación de bits que corresponde a **otro género**, o a un código que no existe (7 bits = 128 códigos para 114 géneros). Opciones: hacer SMOTENC con `track_genre` como texto y codificar *después*, o usar `class_weight`/`scale_pos_weight`, que no crea datos sintéticos (su LightGBM con pesos de clase fue el mejor modelo).

```python
# SMOTENC sobre el género original y luego codificar
smote_nc = SMOTENC(categorical_features=[c in ["track_genre", "key", "mode", "time_signature", "explicit"]
                                         for c in X_train_sin_codificar.columns], random_state=42)
X_res, y_res = smote_nc.fit_resample(X_train_sin_codificar, y_train)
X_res = enc_genero.fit_transform(X_res)   # el género sintético siempre es un género real
```

**4. Sesgo al orientar el marketing.** La respuesta quedó bien. Para completar la parte de sesgo:
- La popularidad refleja lo que Spotify ya promovió, así que el modelo reproduce esas decisiones.
- El codificador de artista favorece a los artistas consolidados: el 24% de los artistas nuevos recibe solo el promedio global.
- Usarlo para decidir a quién promocionar refuerza a los ya conocidos. Conviene reservar presupuesto para artistas emergentes y revisar el desempeño por grupo.
