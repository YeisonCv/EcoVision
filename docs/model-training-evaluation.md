# Entrenamiento y evaluación del modelo

Esta documentación resume los datos del notebook de entrenamiento y la evaluación final reportada para el modelo utilizado por EcoVision. No se reentrenó ni modificó el artefacto para elaborar este documento.

## Dataset

Se utilizó **Garbage Classification**, de Mostafa Abla, con **15.515 imágenes** distribuidas en 12 categorías. La división fue estratificada con semilla `42`.

| Conjunto | Imágenes | Proporción |
|---|---:|---:|
| Entrenamiento | 10.860 | 70 % |
| Validación | 2.327 | 15 % |
| Prueba | 2.328 | 15 % |

Las categorías, en el orden de `classes.txt`, son: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash y white-glass.

## Preprocesamiento y aumento de datos

Las imágenes se prepararon para una entrada RGB de `224 × 224 × 3`. Durante el entrenamiento se aplicaron volteo horizontal, rotación, zoom y variación de contraste.

## Arquitectura del modelo

La base es **MobileNetV2** con pesos preentrenados de ImageNet. La cabeza de clasificación utiliza GlobalAveragePooling2D, BatchNormalization, una capa densa de 256 neuronas, Dropout de `0.4` y una salida softmax de 12 clases. El optimizador es Adam y la función de pérdida es Sparse Categorical Crossentropy.

## Proceso de entrenamiento

| Etapa | Configuración |
|---|---|
| Transfer Learning | Base MobileNetV2 congelada; se entrenaron las capas de clasificación. Máximo 15 épocas y learning rate inicial `1e-3`. |
| Fine-tuning | Se descongelaron las últimas 30 capas, manteniendo congeladas las capas Batch Normalization. Máximo 10 épocas y learning rate `1e-5`. |

Se utilizaron EarlyStopping, ReduceLROnPlateau y ModelCheckpoint para controlar el entrenamiento y guardar checkpoints.

## Evaluación final

Las métricas siguientes corresponden a la ejecución final sobre las **2.328 imágenes del conjunto de prueba**. Se documentan estas cifras oficiales; no se sustituyen por resultados de ejecuciones anteriores.

| Métrica | Resultado |
|---|---:|
| Test Accuracy | 95,27 % |
| Test Loss | 0,1519 |
| Predicciones correctas | 2.218 |
| Predicciones incorrectas | 110 |
| Precision macro | 93,77 % |
| Recall macro | 93,47 % |
| F1-score macro | 93,58 % |
| Precision ponderada | 95,34 % |
| Recall ponderado | 95,27 % |
| F1-score ponderado | 95,28 % |

### Métricas por categoría

| Categoría | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| battery | 93,75 % | 95,07 % | 94,41 % | 142 |
| biological | 99,30 % | 95,95 % | 97,59 % | 148 |
| brown-glass | 96,63 % | 94,51 % | 95,56 % | 91 |
| cardboard | 96,18 % | 94,03 % | 95,09 % | 134 |
| clothes | 98,87 % | 98,50 % | 98,68 % | 799 |
| green-glass | 95,79 % | 96,81 % | 96,30 % | 94 |
| metal | 84,13 % | 92,17 % | 87,97 % | 115 |
| paper | 94,27 % | 93,67 % | 93,97 % | 158 |
| plastic | 89,92 % | 82,31 % | 85,94 % | 130 |
| shoes | 94,50 % | 98,32 % | 96,37 % | 297 |
| trash | 97,00 % | 93,27 % | 95,10 % | 104 |
| white-glass | 84,87 % | 87,07 % | 85,96 % | 116 |

## Matriz de confusión y reporte CSV

No se encontraron `confusion_matrix.png` ni `classification_report.csv` en el repositorio, así que no se incluye una imagen ni un enlace a un archivo inexistente. Para incorporarlos cuando estén disponibles, cópialos a `docs/figures/confusion_matrix.png` y `docs/figures/classification_report.csv`, respectivamente. La matriz podrá insertarse aquí con la ruta relativa `figures/confusion_matrix.png`.

## Análisis de resultados

El mayor F1-score corresponde a **clothes (98,68 %)**. Los F1-score más bajos son **plastic (85,94 %)**, **white-glass (85,96 %)** y **metal (87,97 %)**.

Entre las confusiones observadas con mayor frecuencia están clothes → shoes (12 casos), plastic → white-glass (9 casos) y white-glass → plastic (7 casos). En conjunto, los resultados muestran buen rendimiento en el conjunto de prueba; las confusiones reportadas indican que algunas categorías pueden ser difíciles de distinguir visualmente. Estas observaciones describen los resultados y no establecen causas.
