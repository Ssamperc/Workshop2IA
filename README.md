# Workshop 2 – Machine Learning & Deep Learning Aplicado

**Universidad EAFIT – Introducción a la Inteligencia Artificial (2026-01)**

**Equipo:** David Quintero · David Ruiz · Juan Pablo Duque · Samuel Samper

---

## Descripción General

Este workshop aborda dos problemas supervisados independientes aplicando el ciclo completo de un proyecto de Machine Learning y Deep Learning: desde el análisis del problema y la exploración de datos, pasando por el preprocesamiento y la extracción de características, hasta el entrenamiento, ajuste de hiperparámetros, evaluación crítica de resultados y prueba con muestras artificiales.

| # | Problema | Tipo de tarea | Dataset | Objetivo |
|---|---|---|---|---|
| 1 | Detección de Fatiga Muscular en Ciclismo | Clasificación binaria supervisada | Muscle Fatigue Cycling (HuggingFace) | Clasificar el estado muscular de un ciclista (normal vs. fatiga) a partir de señales EMG |
| 2 | Estimación de Edad a partir de Imágenes Faciales | Regresión supervisada | Faces: Age Detection (Kaggle / UTKFace) | Predecir la edad de una persona a partir de su imagen facial usando una CNN |

---

## Estructura del Repositorio

```
workshop_2/
├── README.md
├── clasificacion/
│   └── clasificacion.ipynb       ← Notebook completo del Problema 1
└── regresion/
    └── regresion.ipynb           ← Notebook completo del Problema 2
```

---

## Instrucciones de Ejecución

Ambos notebooks están diseñados para correr en **Google Colab** de principio a fin ejecutando las celdas en orden.

### Problema 1 – Clasificación

1. Abrir `clasificacion/clasificacion.ipynb` en Google Colab.
2. Ejecutar todas las celdas en orden. El dataset se descarga automáticamente desde HuggingFace mediante la librería `datasets`.
3. No se requiere ninguna credencial ni archivo adicional.

### Problema 2 – Regresión

1. Obtener las credenciales de la API de Kaggle descargando el archivo `kaggle.json` desde [kaggle.com → Settings → API → Create New Token](https://www.kaggle.com/settings).
2. Abrir `regresion/regresion.ipynb` en Google Colab.
3. Ejecutar la **primera celda** (`files.upload()`), que abrirá un selector de archivos — subir el `kaggle.json` cuando se solicite.
4. Ejecutar el resto de las celdas en orden. El dataset se descarga y descomprime automáticamente.

---

## Tecnologías Utilizadas

**Lenguaje:** Python 3 · **Entorno:** Google Colab

| Librería | Uso principal |
|---|---|
| `pandas`, `numpy` | Manipulación y análisis de datos |
| `matplotlib`, `seaborn` | Visualización y EDA |
| `scipy` | Análisis espectral de señales EMG (método de Welch) |
| `scikit-learn` | Modelos clásicos, pipelines, búsqueda de hiperparámetros, métricas |
| `tensorflow` / `keras` | DNN (clasificación) y CNN (regresión) |
| `datasets` (HuggingFace) | Descarga del dataset de fatiga muscular |
| `kaggle` | Descarga del dataset de imágenes faciales |

---

---

# Problema 1 – Clasificación: Detección de Fatiga Muscular en Ciclismo

## Dataset

- **Nombre:** Muscle Fatigue Cycling
- **Fuente:** HuggingFace – [YominE/Muscle_Fatigue_Cycling](https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling)
- **Descripción:** Señales de electromiografía (EMG) registradas en 8 músculos de la pierna dominante de ciclistas realizando sprints en bicicleta. El target original tiene 3 posibles etiquetas (0, 1, 2), que se binarizaron a 0 (condición normal) y 1 (desgaste muscular).
- **Variables:** columna `Time` (temporal), 8 canales EMG numéricos continuos, columna `Target` (binaria objetivo).

---

## Guía de Navegación del Notebook `clasificacion.ipynb`

El notebook está organizado en 15 secciones numeradas. A continuación se describe qué hace y qué responde cada una.

---

### Sección 0 – Instalación

Instalación de dependencias: `datasets`, `seaborn`, `scikit-learn`, `scipy`, `tensorflow`.

---

### Sección 1 – Librerías

Importación de todas las librerías necesarias. Se fija la semilla aleatoria en `42` tanto para NumPy como para TensorFlow, garantizando reproducibilidad.

---

### Sección 2 – Carga del Dataset

El dataset se descarga directamente desde HuggingFace con `load_dataset("YominE/Muscle_Fatigue_Cycling")` y se convierte a DataFrame de pandas. Se imprime la forma original y las columnas disponibles.

---

### Sección 3 – Análisis Preliminar y Preprocesamiento del Target

**¿Qué hace?**
- Binariza el target reemplazando la etiqueta `2` por `1`, dejando el problema como clasificación binaria: `0 = condición normal`, `1 = desgaste muscular`.
- Clasifica todas las variables presentes en el dataset por tipo:
  - `Time`: numérica continua de tipo temporal.
  - Señales EMG (8 canales): numéricas continuas.
  - `Target`: binaria / categórica objetivo.

**¿Por qué se excluye `Time` del modelado?**
El enunciado indica explícitamente que `Time` no debe incluirse como característica porque su dependencia lineal con el target simplificaría artificialmente el problema.

---

### Sección 4 – Frecuencia de Muestreo y Diseño de Ventanas

**¿Qué hace?**
Estima la frecuencia de muestreo del dataset calculando la diferencia temporal mediana entre muestras consecutivas de la columna `Time`. Con `dt ≈ 0.001 s` se obtiene `fs ≈ 1000 Hz`, lo que implica que una ventana de 1 segundo corresponde exactamente a **1000 muestras**.

Este paso evita asumir arbitrariamente el tamaño de ventana y justifica formalmente el parámetro `window_size = 1000` usado en el feature engineering.

---

### Sección 5 – EDA sobre las Señales Crudas

**¿Qué contiene?**
- Gráfico de balance de clases del target binarizado (con porcentajes por clase).
- Estadísticos descriptivos generales del dataset original.
- Visualización de una porción de 2 segundos (2000 muestras) de los 8 canales EMG.
- Distribuciones (histogramas + KDE) de 4 canales representativos.

**Conclusiones registradas en el notebook:**
- Las señales son oscilatorias y están centradas alrededor de cero, consistente con señales musculares.
- La amplitud y variabilidad varían entre músculos, sugiriendo que cada canal aporta información distinta.
- Existe desbalance de clases: la clase `0` (normal) es mayoritaria, lo que justifica el uso de métricas como Precision, Recall y F1-score en lugar de depender únicamente de Accuracy.
- No es trivial separar fatiga vs. normalidad a simple vista sobre la señal cruda, lo que justifica el feature engineering.

---

### Sección 6 – Feature Engineering (Extracción de Características)

**¿Qué hace?**
Diseña e implementa un algoritmo de ventanas deslizantes no solapadas de 1 segundo (1000 muestras) sobre los 8 canales EMG. Por cada ventana y canal se extraen **6 características** (superando el mínimo de 4 requerido):

| Característica | Dominio | Justificación |
|---|---|---|
| `rms` | Tiempo | Mide la energía efectiva de la contracción muscular |
| `variance` | Tiempo | Refleja la dispersión/amplitud de la señal |
| `zero_crossings` | Tiempo | Resume oscilaciones y cambios de signo, relacionados con actividad neuromuscular |
| `mean_slope` | Tiempo | Cuantifica la variación promedio entre muestras consecutivas |
| `mean_freq` | Frecuencia | Detecta desplazamientos del centro de masa espectral |
| `median_freq` | Frecuencia | Característica clásica de EMG para estudiar fatiga muscular |

Las frecuencias se estiman con el método de Welch (`scipy.signal.welch`). La nueva base de datos resultante tiene la forma `(n_ventanas, 48 características + 1 target)`. El target de cada ventana corresponde a la moda del target en ese segundo.

---

### Sección 7 – EDA Completo sobre la Base de Características

**¿Qué contiene?**
- Revisión de valores nulos en la base transformada.
- Estadísticos descriptivos de todas las características.
- Balance de clases en la base por ventanas.
- Distribuciones de características representativas (histogramas + KDE).
- Heatmap de correlación de todas las características.
- Top 10 características con mayor correlación absoluta con el target.
- Boxplots por clase (target = 0 vs. target = 1) de las 6 características más correlacionadas.

**Interpretaciones clave:**
- El desbalance de clases persiste en la base transformada.
- Se observan grupos de variables correlacionadas entre características del mismo músculo (redundancia parcial esperada).
- Los boxplots muestran diferencias de mediana y dispersión entre clases en varias características, confirmando que existe señal discriminativa útil.

---

### Sección 8 – Procesamiento de Datos

**¿Qué hace?**
- Divide los datos en `X_train / X_val / X_test` con proporciones **70 / 15 / 15** mediante `train_test_split` estratificado (preserva la proporción de clases en cada subconjunto).
- Construye un **pipeline reproducible con scikit-learn** (`ColumnTransformer`) que incluye:
  - `SimpleImputer(strategy="median")` para manejo de valores faltantes.
  - `StandardScaler` para estandarización de características numéricas.

**Justificación del split:**
El conjunto de validación permite comparar y ajustar hiperparámetros sin contaminar la evaluación final; el conjunto de prueba queda reservado para la estimación objetiva del desempeño real del modelo seleccionado.

---

### Sección 9 – Entrenamiento y Ajuste de Hiperparámetros (Modelos Clásicos)

**¿Qué hace?**
Entrena y ajusta hiperparámetros de los 4 modelos clásicos usando `GridSearchCV` con validación cruzada estratificada de 3 folds (`StratifiedKFold`) y optimizando `f1_weighted`:

| Modelo | Hiperparámetros explorados |
|---|---|
| k-Nearest Neighbors | `n_neighbors` ∈ {3,5,7,9,11}, `weights` ∈ {uniform, distance} |
| Decision Tree | `max_depth` ∈ {3,5,7,10,None}, `min_samples_split` ∈ {2,5,10}, `min_samples_leaf` ∈ {1,2,4} |
| Random Forest | `n_estimators` ∈ {100,200}, `max_depth` ∈ {5,10,15,None}, `min_samples_split`, `min_samples_leaf` |
| Gradient Boosting | `n_estimators` ∈ {50,100,150}, `learning_rate` ∈ {0.01,0.05,0.1}, `max_depth` ∈ {2,3,4} |

Al final se genera una **tabla comparativa** con Accuracy, Precision, Recall y F1-Score sobre `X_train`, `X_val` y `X_test` para los 4 modelos.

---

### Sección 10 – Curvas de Entrenamiento / Validación (Modelos Clásicos)

Para cada uno de los 4 mejores modelos entrenados se grafica la **learning curve** (F1-weighted vs. tamaño del conjunto de entrenamiento), permitiendo detectar visualmente overfitting o underfitting:

- Brecha grande entre curva de train y curva de val → **overfitting**.
- Ambas curvas bajas → **underfitting**.
- Convergencia en valores altos → buen balance.

---

### Sección 11 – Deep Neural Network (DNN)

**Arquitectura implementada:**

```
Input → Dense(128, ReLU) → Dropout(0.3)
      → Dense(64, ReLU)  → Dropout(0.3)
      → Dense(32, ReLU)  → Dropout(0.2)
      → Dense(1, Sigmoid)
```

- Optimizer: `Adam(lr=0.001)`
- Loss: `binary_crossentropy`
- Callback: `EarlyStopping(patience=5, restore_best_weights=True)`
- Escalado propio con `StandardScaler` (independiente del pipeline de modelos clásicos).

Se reportan métricas sobre train, val y test, y se grafican las curvas de pérdida y accuracy por época para análisis de overfitting.

Las métricas de la DNN se agregan a la tabla comparativa de la Sección 9, completando la comparación entre los **5 modelos**.

---

### Sección 12 – Selección del Mejor Modelo

Se selecciona automáticamente el modelo con mayor `Test F1` (desempate por `Val F1`). Se responden las preguntas del enunciado:

- ¿Cuál modelo tuvo mejor desempeño?
- ¿Alguno presentó overfitting o underfitting? ¿Cómo se detectó?
- ¿Cuál se seleccionaría para producción y por qué?

---

### Sección 13 – Reentrenamiento Final y Evaluación sobre `X_test`

**¿Qué hace?**
El mejor modelo se reentrena con `X_train + X_val` (combinados) y se evalúa sobre `X_test`. Esto maximiza los datos de entrenamiento sin comprometer la objetividad de la evaluación final.

Si el mejor modelo es la DNN, se reconstruye la arquitectura y se reentrena por el mismo número de épocas que el entrenamiento anterior.

**Resultados reportados:**
- Tabla de métricas finales (Accuracy, Precision, Recall, F1).
- `classification_report` completo por clase.
- **Matriz de confusión** (visualizada con heatmap).
- **Boxplots** de las 4 características más correlacionadas con el target, diferenciando muestras de cada clase.

**Interpretación de la matriz de confusión:** Los falsos negativos (fatiga no detectada) son especialmente relevantes en este contexto, lo que justifica priorizar Recall y F1 sobre Accuracy.

---

### Sección 14 – Prueba con Muestra Artificial

Se genera una muestra artificial con valores plausibles construida a partir de la media y desviación estándar de la base de entrenamiento (`X_trainval`). Para cada característica se muestrea un valor de una distribución normal `N(μ, σ)`.

La muestra se ingresa al modelo final y se reporta si se clasifica como **fatigado** o **no fatigado**, con análisis de si el resultado tiene sentido en el contexto del problema.

---

### Sección 15 – Resumen Final

Checklist automático que confirma que todas las secciones del enunciado fueron cumplidas, seguido de celdas Markdown con la narrativa explicativa completa del notebook (análisis de características, EDA, pipeline, modelos, selección, evaluación y conclusiones).

---

---

# Problema 2 – Regresión: Estimación de Edad a partir de Imágenes Faciales

## Dataset

- **Nombre:** Faces: Age Detection from Images (UTKFace)
- **Fuente:** Kaggle – [arashnic/faces-age-detection-dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset)
- **Descripción:** Imágenes faciales etiquetadas con la edad del sujeto. La edad está codificada en el nombre de archivo (`<age>_<gender>_<ethnicity>_<timestamp>.jpg`). El objetivo es predecir la edad como valor continuo a partir de los píxeles de la imagen.
- **Subconjunto utilizado:** Se trabaja con **3000 imágenes** (muestreadas aleatoriamente) para optimizar el tiempo de entrenamiento sin sacrificar representatividad.

---

## Guía de Navegación del Notebook `regresion.ipynb`

El notebook está compuesto por 4 celdas (las primeras 3 son de configuración de Kaggle). La celda principal está organizada en 11 secciones numeradas.

---

### Celdas 1–3 – Configuración de Kaggle y Descarga del Dataset

- **Celda 1:** `files.upload()` — permite subir el archivo `kaggle.json` con las credenciales de la API.
- **Celda 2:** Copia `kaggle.json` a `~/.kaggle/` y ajusta los permisos.
- **Celda 3:** Descarga y descomprime el dataset con `kaggle datasets download`.

---

### Sección 0–1 – Instalación e Imports

Instalación de `kaggle`, `tensorflow`, `matplotlib`, `seaborn`, `pandas`, `scikit-learn`. Importación de librerías y fijación de semilla aleatoria `42`.

---

### Sección 3 – Carga del Dataset

Se recorre el directorio de imágenes descargadas con `os.walk`, extrayendo la edad desde el nombre del archivo mediante expresión regular `r"^(\d+)"`. Se construye un DataFrame con columnas `filename` y `age`, del cual se muestrea un subconjunto de **3000 imágenes** con `random_state=42`.

Se imprime el rango de edades, la media y las primeras filas del DataFrame.

---

### Sección 4 – Análisis Preliminar del Problema

**¿Por qué es un problema de regresión?**
1. La variable objetivo (`age`) es numérica continua.
2. El objetivo es predecir un valor numérico, no una categoría discreta.
3. Las métricas de evaluación son MAE, RMSE y R².
4. La función de activación de salida de la CNN es **lineal**.

**Características de entrada documentadas:**
- Tipo: imágenes RGB (3 canales).
- Dimensiones: redimensionadas a **128 × 128 píxeles**.
- Espacio de color: RGB, valores originales 0–255 normalizados a [0, 1].

**Protocolo de adquisición (UTKFace):**
- Imágenes faciales de personas de 0 a 116 años, con diversidad de género y etnia.
- Posibles sesgos: iluminación variable, orientación del rostro, accesorios.

---

### Sección 5 – EDA Completo

**¿Qué contiene?**

**Distribución de edades:**
- Histograma con líneas de media y mediana superpuestas.
- Boxplot de la distribución de edades.
- Estadísticos descriptivos (mínimo, máximo, media, mediana, desviación estándar).

**Análisis de balance por rangos:**
La distribución se analiza en 5 grupos etarios:

| Rango | Grupo |
|---|---|
| 0–12 años | Niños |
| 13–25 años | Jóvenes |
| 26–40 años | Adultos jóvenes |
| 41–60 años | Adultos |
| 61+ años | Adultos mayores |

Se grafica un gráfico de barras por rango. La distribución está sesgada hacia edades jóvenes (20–30 años), lo que representa un sesgo potencial del modelo.

**Visualización de muestras representativas:**
Se muestra 1 imagen de muestra de cada uno de los 5 grupos etarios con su edad real, permitiendo una inspección visual de la calidad y variabilidad del dataset.

---

### Sección 6 – Procesamiento de Datos

**División estratificada 70/15/15:**
Se crea una columna auxiliar `age_group` y se realiza un split estratificado por grupo etario para asegurar que cada subconjunto (train, val, test) tenga una distribución representativa de todas las edades.

**Data Augmentation (solo sobre entrenamiento):**
Se aplican las siguientes transformaciones con `ImageDataGenerator`:

| Transformación | Parámetro |
|---|---|
| Rotación | ±15° |
| Desplazamiento horizontal | ±10% |
| Desplazamiento vertical | ±10% |
| Flip horizontal | Sí |
| Zoom | ±10% |

El conjunto de validación y prueba usa únicamente `rescale=1./255` sin aumentación.

**Generadores de Keras:**
Se crean generadores `flow_from_dataframe` para train, val y test con `target_size=(128, 128)` y `batch_size=32`. El generador de test tiene `shuffle=False` para preservar el orden de las predicciones.

---

### Sección 7 – Arquitectura CNN

La red implementada tiene 3 bloques convolucionales seguidos de 3 capas densas ocultas:

```
Input (128×128×3)
  │
  ├── Conv2D(16, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.2)
  ├── Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.2)
  ├── Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.2)
  │
  └── Flatten
       ├── Dense(64, ReLU) → Dropout(0.3)
       ├── Dense(32, ReLU) → Dropout(0.3)
       ├── Dense(16, ReLU)
       └── Dense(1, lineal)      ← Salida de regresión
```

**Justificación de decisiones de diseño:**

| Elemento | Justificación |
|---|---|
| Progresión 16→32→64 filtros | Extracción de características de baja a alta complejidad progresivamente |
| BatchNormalization | Estabiliza y acelera el entrenamiento |
| Dropout (0.2–0.3) | Regularización para reducir overfitting |
| Activación lineal en salida | Imprescindible para regresión sobre valores continuos |
| MSE como función de pérdida | Penaliza errores grandes, adecuado para estimación de edad |
| ~180,000 parámetros totales | Arquitectura ligera que entrena en minutos en Colab |

**Compilación y callbacks:**
- Optimizer: `Adam(lr=0.001)`
- Loss: `mse`
- Callbacks: `EarlyStopping(patience=5, restore_best_weights=True)` y `ReduceLROnPlateau(factor=0.5, patience=3)`

---

### Sección 8 – Entrenamiento

Entrenamiento por hasta **20 épocas**, con parada temprana automática según `val_loss`. Se usa el generador de train para data augmentation en tiempo real.

---

### Sección 9 – Curvas de Pérdida

Se grafican en paralelo:
- Curvas de **MSE (Loss)** en train y val por época.
- Curvas de **MAE** en train y val por época.

Permiten detectar overfitting (divergencia entre curvas) o underfitting (ambas curvas altas).

---

### Sección 10 – Evaluación Final

**Métricas reportadas sobre `X_test`:**

| Métrica | Descripción |
|---|---|
| MAE | Error absoluto promedio en años |
| RMSE | Penaliza errores grandes (raíz del error cuadrático medio) |
| R² | Proporción de varianza explicada por el modelo |

Se incluye un **gráfico de dispersión predicciones vs. valores reales** con la línea de referencia perfecta (`y = x`), que permite visualizar la concentración de errores por rango de edad.

Se presenta una tabla final que combina el valor numérico de cada métrica con su interpretación en lenguaje natural (ej. "Error promedio de X años", "Explica Y% de la varianza").

---

### Sección 11 – Prueba con Variaciones de Imagen

Se toma la primera imagen del conjunto de prueba y se le aplican 4 variaciones para analizar la sensibilidad del modelo:

| Variación | Descripción |
|---|---|
| Original | Imagen sin modificar |
| Brillo +50% | Píxeles multiplicados por 1.5 (con clip a 1.0) |
| Brillo −50% | Píxeles multiplicados por 0.5 |
| Flip horizontal | Espejo horizontal de la imagen |

Se muestran las 4 variantes con su predicción y la edad real, y se reporta el error de la predicción original. El análisis discute cómo cambios en iluminación, escala u orientación pueden afectar las predicciones del modelo.

---

## Criterios de Evaluación

### Problema 1 – Clasificación

| Criterio | Peso | Secciones del notebook |
|---|---|---|
| Justificación teórica y análisis preliminar | 20% | Secciones 3, 4 |
| Calidad del EDA, extracción de características e interpretaciones | 20% | Secciones 5, 6, 7 |
| Preprocesamiento de datos y pipeline | 20% | Sección 8 |
| Implementación, ajuste de hiperparámetros y comparación de modelos | 25% | Secciones 9, 10, 11, 12 |
| Prueba con muestra artificial y análisis | 15% | Sección 14 |

### Problema 2 – Regresión

| Criterio | Peso | Secciones del notebook |
|---|---|---|
| Justificación teórica y análisis preliminar | 20% | Sección 4 |
| Calidad del EDA e interpretaciones | 20% | Sección 5 |
| Preprocesamiento de datos y pipeline | 20% | Sección 6 |
| Implementación y evaluación del modelo CNN | 25% | Secciones 7, 8, 9, 10 |
| Prueba con muestra artificial y análisis | 15% | Sección 11 |
