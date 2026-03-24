# Workshop 2 – Machine Learning & Deep Learning Aplicado

**Universidad EAFIT – Introducción a la Inteligencia Artificial (2026-01)**

**Equipo:** David Quintero · David Ruiz · Juan Pablo Duque · Samuel Samper

---

## Descripción General

Este workshop integra dos problemas supervisados independientes aplicando el ciclo completo de un proyecto de Machine Learning y Deep Learning: análisis del problema, exploración de datos, preprocesamiento, entrenamiento, evaluación y análisis crítico de resultados.

| Problema | Tipo | Dataset | Objetivo |
|---|---|---|---|
| 1 – Clasificación | Binaria supervisada | Muscle Fatigue Cycling (HuggingFace) | Detectar fatiga muscular en ciclistas a partir de señales EMG |
| 2 – Regresión | Regresión supervisada | Faces: Age Detection (Kaggle) | Estimar la edad de una persona a partir de su imagen facial |

---

## Estructura del Repositorio

```
workshop_2/
├── README.md
├── clasificacion/
│   └── clasificacion.ipynb
└── regresion/
    └── regresion.ipynb
```

---

## Problema 1 – Clasificación: Detección de Fatiga Muscular en Ciclismo

### Dataset

- **Nombre:** Muscle Fatigue Cycling
- **Fuente:** HuggingFace – [YominE/Muscle_Fatigue_Cycling](https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling)
- **Descripción:** Señales de electromiografía (EMG) registradas en 8 músculos de la pierna dominante de ciclistas realizando sprints. La variable objetivo indica el estado muscular: condición normal vs. desgaste muscular.
- **Carga:** El dataset se descarga automáticamente desde HuggingFace usando la librería `datasets`.

### Desarrollo del Notebook

El notebook `clasificacion/clasificacion.ipynb` cubre el flujo completo en 15 secciones:

#### 1. Análisis Preliminar

Se preprocesa el target estableciendo 2 etiquetas binarias (`0` = condición normal, `1` = desgaste muscular), reemplazando la etiqueta original `2` por `1`. Se clasifican todas las variables del dataset: las 8 señales EMG son variables numéricas continuas, `Time` es una variable temporal continua (excluida del modelado por su dependencia lineal con el target), y `Target` es la variable binaria objetivo.

#### 2. Estimación de Frecuencia de Muestreo y Ventanas

Se estima la frecuencia de muestreo calculando la diferencia temporal mediana entre muestras consecutivas de la columna `Time`. Con `dt ≈ 0.001 s`, se obtiene `fs ≈ 1000 Hz`, lo que equivale a ventanas de **1000 muestras por segundo**.

#### 3. Feature Engineering (Extracción de Características)

Se diseña un algoritmo de ventanas deslizantes de 1 segundo sobre los 8 canales EMG. Por cada ventana y canal se extraen **mínimo 4 características**, combinando:

- **Dominio del tiempo:** RMS, varianza, cruce por cero, pendiente media de la señal.
- **Dominio de la frecuencia:** Frecuencia mediana, frecuencia media, potencia espectral (via Welch).

Cada característica se justifica por su relevancia en la detección de fatiga muscular. La variable `Time` es excluida de la base final.

#### 4. EDA Completo

Se realiza análisis exploratorio sobre la base de características extraídas:

- Distribuciones de variables y estadísticos descriptivos.
- Heatmap de correlación entre características.
- Análisis de separabilidad con boxplots por clase.
- Balance de clases y evaluación del desbalance (relevante para la elección de métricas).
- Visualización de una porción de las señales crudas en el tiempo.

#### 5. Preprocesamiento y Pipeline

- Manejo de valores nulos con `SimpleImputer`.
- Estandarización de características con `StandardScaler`.
- Pipeline reproducible con `scikit-learn`.
- División estratificada en `X_train / X_val / X_test` con proporciones **70/15/15**.

#### 6. Entrenamiento y Comparación de Modelos

Se entrenan y comparan los siguientes clasificadores con ajuste de hiperparámetros mediante **Grid Search** o **Random Search**:

| Modelo | Estrategia de búsqueda |
|---|---|
| k-Nearest Neighbors (kNN) | GridSearchCV |
| Decision Tree | GridSearchCV |
| Random Forest | RandomizedSearchCV |
| Gradient Boosting | RandomizedSearchCV |
| Deep Neural Network (DNN) | Arquitectura manual (≥ 3 capas ocultas, Dropout, EarlyStopping) |

Se presenta una **tabla comparativa** con Accuracy, Precision, Recall y F1-Score sobre `X_train`, `X_val` y `X_test`, junto con curvas de aprendizaje para detectar overfitting o underfitting.

#### 7. Evaluación Final del Mejor Modelo

El mejor modelo se reentrena con `X_train + X_val` y se evalúa sobre `X_test`. Se reportan métricas finales, matriz de confusión, y boxplots de características representativas diferenciando muestras fatigadas vs. normales.

#### 8. Prueba con Muestra Artificial

Se genera una muestra aleatoria con valores aproximados a la distribución real (usando media y desviación estándar de la base transformada) y se predice su clase. Se analiza si el resultado tiene sentido en el contexto del problema.

---

## Problema 2 – Regresión: Estimación de Edad a partir de Imágenes Faciales

### Dataset

- **Nombre:** Faces: Age Detection from Images
- **Fuente:** Kaggle – [arashnic/faces-age-detection-dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset)
- **Descripción:** Imágenes faciales etiquetadas con la edad del sujeto. El objetivo es entrenar un modelo CNN que estime la edad como variable continua a partir de los píxeles.
- **Carga:** El dataset se descarga desde Kaggle usando la API de Kaggle (`kaggle.json` requerido).

### Desarrollo del Notebook

El notebook `regresion/regresion.ipynb` cubre el flujo completo en 11 secciones:

#### 1. Análisis Preliminar

Se justifica por qué este es un problema de **regresión**: la variable objetivo (edad) es continua. Se describen las características de entrada (dimensiones de imagen, espacio de color, distribución de edades). Se trabaja con un subconjunto de **3000 imágenes** para optimizar el tiempo de entrenamiento sin sacrificar representatividad.

#### 2. EDA Completo

- Histograma y estadísticos descriptivos de la distribución de edades.
- Análisis de balance y sesgos en la distribución del target.
- Visualización de muestras representativas del dataset.
- Análisis de calidad y variabilidad de las imágenes.

#### 3. Preprocesamiento

- Redimensionamiento y normalización de imágenes a escala `[0, 1]`.
- Estrategia de **data augmentation** (rotaciones, flips, zoom).
- División estratificada en `X_train / X_val / X_test` con proporciones **70/15/15**.
- Pipeline reproducible basado en generadores de Keras.

#### 4. Modelo CNN para Regresión

Se entrena una CNN ligera que cumple todos los requisitos del enunciado:

- Capas convolucionales + MaxPooling.
- Batch Normalization y Dropout para regularización.
- Capa de salida con activación lineal (regresión).
- Función de pérdida: **MAE** o **Huber Loss**.

#### 5. Evaluación y Curvas de Pérdida

Se reportan métricas sobre `X_train`, `X_val` y `X_test`:

| Métrica | Descripción |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coeficiente de determinación |

Se grafican curvas de pérdida por época y se analiza si hay overfitting o underfitting. Se incluye gráfico de predicciones vs. valores reales.

#### 6. Prueba con Muestra Artificial

Se toma una imagen de prueba (mismas dimensiones y canales que el entrenamiento) y se analiza la predicción del modelo, discutiendo qué ocurriría al modificar características visuales como iluminación, escala u orientación.

---

## Tecnologías Utilizadas

**Lenguaje:** Python 3

**Librerías principales:**

- `pandas`, `numpy` – Manipulación de datos
- `matplotlib`, `seaborn` – Visualización
- `scikit-learn` – Modelos clásicos, pipelines, métricas, búsqueda de hiperparámetros
- `tensorflow` / `keras` – DNN (clasificación) y CNN (regresión)
- `scipy` – Análisis espectral (Welch) para feature engineering EMG
- `datasets` (HuggingFace) – Carga del dataset de fatiga muscular
- `kaggle` – Descarga del dataset de imágenes faciales

**Entorno:** Google Colab

---

## Instrucciones de Ejecución

### Problema 1 – Clasificación

1. Abrir `clasificacion/clasificacion.ipynb` en Google Colab.
2. Ejecutar todas las celdas en orden. El dataset se descarga automáticamente desde HuggingFace.

### Problema 2 – Regresión

1. Obtener credenciales de Kaggle (`kaggle.json`) desde [kaggle.com/settings](https://www.kaggle.com/settings).
2. Abrir `regresion/regresion.ipynb` en Google Colab.
3. En la primera celda, subir el archivo `kaggle.json` cuando se solicite.
4. Ejecutar todas las celdas en orden. El dataset se descarga y descomprime automáticamente.

---
