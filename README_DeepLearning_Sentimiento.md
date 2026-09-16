# Clasificación de Sentimiento de Comentarios con Deep Learning (NLP)

Modelo de red neuronal con **TensorFlow/Keras** para clasificar comentarios de clientes como positivos o negativos, construido en **dos variantes deliberadas** sobre el mismo dataset para exponer y justificar el efecto de una decisión de preprocesamiento: qué hacer con los datos duplicados.

## 🎯 Problema

A partir de un dataset de 1.000 comentarios de clientes etiquetados como positivo/negativo, el objetivo es entrenar un modelo de Deep Learning capaz de clasificar automáticamente el sentimiento de un comentario nuevo, usando técnicas de procesamiento de lenguaje natural (Tokenizer, Embedding) en lugar de reglas manuales de palabras clave.

## 🧪 Por qué dos versiones del modelo (y no una)

Al inspeccionar el dataset se encontró que, de 1.000 comentarios, **980 eran duplicados exactos** (solo 20 comentarios únicos, algunos repetidos hasta 59 veces como frases plantilla). Esto plantea un dilema real de ciencia de datos, no cosmético: eliminar duplicados es buena práctica para evitar fuga de datos entre entrenamiento y prueba, pero aquí dejaba solo 20 registros — insuficiente para entrenar una red neuronal de forma confiable.

En vez de elegir una sola opción y ocultar el trade-off, el proyecto implementa y documenta **ambas**, con su justificación:

- **Opción A — duplicados eliminados:** dataset reducido a 20 registros únicos, evitando cualquier fuga de datos, pero con muestra insuficiente para un entrenamiento robusto.
- **Opción B — duplicados conservados:** dataset completo de 1.000 registros, justificado porque las repeticiones son frases plantilla legítimas del negocio (no errores de carga), lo que permite un entrenamiento con más ejemplos — a costa de que el mismo texto pueda aparecer tanto en entrenamiento como en prueba.

## 🛠️ Stack y metodología

- **Framework:** TensorFlow / Keras
- **Preprocesamiento de texto:** `Tokenizer` (vocabulario máximo 1.000 palabras, token `<OOV>` para palabras desconocidas) + `pad_sequences` para longitud uniforme
- **Arquitectura:** `Embedding` (32 dimensiones) → `GlobalAveragePooling1D` → `Dense(16, relu)` → `Dropout(0.3)` → `Dense(1, sigmoid)` para clasificación binaria
- **División de datos:** 70% entrenamiento / 15% validación / 15% prueba, estratificada
- **Entrenamiento:** `Adam`, `binary_crossentropy`, `EarlyStopping` (paciencia 10) monitoreando `val_loss`
- **Validación con datos nunca vistos:** además de la métrica de test, el modelo se prueba con comentarios nuevos escritos a mano, para verificar que generaliza más allá del propio dataset
- **Persistencia:** el modelo entrenado se guarda en formato `.keras` (`model.save()`) para reutilización posterior

## 📊 Resultados

| Variante | Registros | Train / Val / Test | Accuracy (test) |
|---|---|---|---|
| Opción A (sin duplicados) | 20 | 14 / 3 / 3 | 66.67% |
| Opción B (con duplicados) | 1.000 | 700 / 150 / 150 | 100.00% |

**Lectura honesta de estos números:** el 100% de la Opción B no debe leerse como "el mejor modelo" sin más — con solo 20 frases distintas repetidas cientos de veces, es esperable que el mismo texto (o uno casi idéntico) aparezca en entrenamiento y en prueba, inflando la métrica. El 66.67% de la Opción A, con una muestra de apenas 20 registros, tampoco es representativo del verdadero desempeño del modelo. Ambos resultados se presentan juntos precisamente para exponer esta tensión entre calidad de la validación y tamaño de muestra, en lugar de esconder la limitación detrás de un solo número.

## 🚀 Cómo ejecutarlo

```bash
# Clonar el repositorio
git clone <https://github.com/haroldrodriguezadm-png/Clasificaci-n-de-Sentimiento-de-Comentarios-con-Deep-Learning-NLP-/blob/main/README_DeepLearning_Sentimiento.md>
cd <Clasificación de Sentimiento de Comentarios con Deep Learning (NLP)>

# Instalar dependencias
pip install tensorflow scikit-learn pandas numpy openpyxl

# Ejecutar el notebook
jupyter notebook "Aplicaciones reales del Deep Learning.ipynb"
```

El notebook espera el archivo de datos en la ruta indicada en `RUTA_ARCHIVO` (por defecto apunta a una ruta de Google Colab). Ajústala a la ubicación local antes de ejecutar.

**Requisitos:** Python 3.9+, TensorFlow 2.x

## 📁 Estructura del repositorio

```
├── Aplicaciones reales del Deep Learning.ipynb   # Notebook principal (Opción A y B)
├── data/
│   └── comentarios_clientes.xlsx                 # Dataset de entrada
├── modelo_sentimiento_clientes.keras             # Modelo entrenado (Opción A)
└── README.md
```

## 🔭 Limitaciones y próximos pasos

- El dataset real solo contiene **20 patrones de texto distintos**; ninguna de las dos métricas refleja el desempeño esperado sobre comentarios verdaderamente nuevos y variados en producción.
- Próximo paso natural: conseguir un dataset con mayor diversidad léxica real (no frases plantilla repetidas), lo que permitiría eliminar duplicados sin sacrificar volumen de entrenamiento.
- Con más datos, valdría la pena escalar la arquitectura hacia capas recurrentes (LSTM/GRU) o embeddings preentrenados, que el propio notebook identifica como el camino de evolución natural de este enfoque.

## 👤 Autor

Harold Rodríguez B. — [LinkedIn](https://www.linkedin.com/in/harold-rodriguez-boisset/) 
