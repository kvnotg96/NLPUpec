# Entrenamiento eficiente en memoria de redes neuronales mediante programación no lineal

![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Estado](https://img.shields.io/badge/estado-investigación%20científica-green)
![Versión](https://img.shields.io/badge/versión-v1.0.0-orange)
[![Repositorio](https://img.shields.io/badge/repositorio-GitHub-black)](https://github.com/kvnotg96/NLPUpec)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20128567.svg)](https://doi.org/10.5281/zenodo.20128567)

Este repositorio contiene el código fuente, los scripts experimentales, las métricas numéricas, las figuras generadas y los archivos de reproducibilidad asociados al artículo científico **“Programación no lineal con restricciones enteras para el entrenamiento eficiente en memoria de redes neuronales”**.

El estudio propone y evalúa un enfoque de optimización para el entrenamiento de redes neuronales artificiales orientado a reducir la complejidad paramétrica del modelo y mejorar su eficiencia en memoria, sin comprometer de forma significativa el desempeño predictivo. Para ello, se formula un esquema de optimización jerárquico que combina entrenamiento neuronal, regularización de pesos y ajuste de hiperparámetros mediante programación no lineal.

Desde el punto de vista computacional, el repositorio implementa una red neuronal multicapa tipo perceptrón multicapa, con dos capas ocultas, funciones de activación configurables, entrenamiento mediante Adam, regularización L2, parada temprana y ajuste automático de la tasa de aprendizaje. Sobre esta base se incorpora una etapa de optimización externa mediante el método **L-BFGS-B**, orientada a estimar los hiperparámetros asociados a la regularización y al aprendizaje.

Los experimentos se desarrollan sobre cinco conjuntos de datos sintéticos bidimensionales y balanceados: **Espiral, Lunas, Círculos, XOR y Gaussianas**. Para cada conjunto de datos se compara una red neuronal base frente a una red neuronal optimizada, evaluando pérdida, precisión, norma de pesos, reducción relativa de la norma, número de evaluaciones del optimizador y evolución de la función objetivo.

El propósito del repositorio es facilitar la trazabilidad, revisión y reproducción de los resultados computacionales reportados en el artículo.

---

## Descripción científica

El marco computacional implementado en este repositorio sigue una perspectiva de optimización jerárquica para el entrenamiento eficiente en memoria de redes neuronales. En el nivel inferior se entrena un perceptrón multicapa mediante la minimización de la pérdida de entropía cruzada binaria con regularización L2. En el nivel superior se aplica un procedimiento de optimización no lineal para ajustar la tasa de aprendizaje y el coeficiente de regularización, con el objetivo de obtener una red neuronal más compacta y con menor norma de pesos.

El problema se interpreta como un equilibrio entre desempeño predictivo y eficiencia paramétrica. El modelo base prioriza la precisión de clasificación, mientras que el modelo optimizado incorpora un criterio adicional asociado a la reducción de la magnitud de los parámetros entrenables. En este contexto, la norma euclidiana de los pesos se utiliza como indicador de eficiencia paramétrica y como aproximación computacional al criterio de memoria.

El diseño experimental incluye cinco problemas sintéticos de clasificación binaria con diferentes geometrías: separación en espiral, lunas, círculos concéntricos, patrón XOR y agrupamientos gaussianos. Estos conjuntos permiten evaluar el comportamiento del método ante distintos niveles de separabilidad no lineal.

El repositorio incluye una versión interactiva del módulo experimental, un script de ejecución sin interfaz gráfica, una versión portable para reproducción en GitHub, métricas numéricas exportadas en formato JSON y archivos de metadatos para citación y preservación del software.

---

## Estructura del repositorio

```text
entrenamiento-eficiente-memoria-redes-neuronales/
├── README.md
├── CITATION.cff
├── codemeta.json
├── .zenodo.json
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
│   ├── neural_opt.py
│   ├── run_experiments.py
│   └── run_experiments_portable.py
├── results/
│   └── metrics.json


```

---

## Archivos principales

### `src/neural_opt.py`

Módulo experimental interactivo que implementa la interfaz gráfica, la generación de conjuntos de datos, el entrenamiento del perceptrón multicapa, la optimización no lineal de hiperparámetros y la interpretación automática de resultados. Permite seleccionar el conjunto de datos, la función de activación, la arquitectura de la red, los parámetros de entrenamiento y los criterios de optimización orientados a memoria.

### `src/run_experiments.py`

Script experimental sin interfaz gráfica utilizado para reproducir los resultados numéricos y generar las figuras asociadas al artículo. Ejecuta el flujo completo de experimentación sobre los cinco conjuntos de datos, entrena la red base y la red optimizada, aplica L-BFGS-B para el ajuste de hiperparámetros, almacena las métricas finales y exporta las figuras.

### `src/run_experiments_portable.py`

Versión portable del script experimental, adaptada para su ejecución dentro del repositorio. A diferencia del script original, evita rutas absolutas locales y guarda las figuras generadas en la carpeta relativa `figures/`.

### `results/metrics.json`

Archivo estructurado que contiene los resultados numéricos finales para cada conjunto de datos. Incluye pérdida de la red base y optimizada, precisión, norma de pesos, reducción relativa de la norma, coeficiente óptimo de regularización, tasa óptima de aprendizaje, número de evaluaciones del optimizador y valores iniciales y finales de la función objetivo.

### `article/article-kevin.tex`

Archivo fuente en LaTeX del artículo científico asociado al repositorio.

---

## Requisitos

El repositorio fue desarrollado en Python y requiere las siguientes bibliotecas principales:

```text
numpy
scipy
matplotlib
scikit-learn
```

Para instalar todas las dependencias, se recomienda utilizar el archivo `requirements.txt`.

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/USUARIO/entrenamiento-eficiente-memoria-redes-neuronales.git
cd entrenamiento-eficiente-memoria-redes-neuronales
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el módulo interactivo

```bash
python src/neural_opt.py
```

### 5. Ejecutar los experimentos reproducibles

```bash
python src/run_experiments_portable.py
```

Las figuras se guardarán en la carpeta:

```text
figures/
```

Las métricas finales se almacenarán en:

```text
results/metrics.json
```

---

## Resultados reproducibles

Los resultados experimentales muestran que el procedimiento de optimización propuesto permite reducir la norma de los pesos de la red neuronal, manteniendo un desempeño de clasificación competitivo en los cinco conjuntos de datos evaluados.

| Conjunto de datos | Precisión base | Precisión optimizada | Reducción de la norma de pesos |
|---|---:|---:|---:|
| Espiral | 98.00% | 94.00% | 27.82% |
| Lunas | 100.00% | 100.00% | 43.66% |
| Círculos | 100.00% | 100.00% | 43.36% |
| XOR | 97.00% | 96.00% | 27.17% |
| Gaussianas | 99.75% | 99.50% | 6.39% |

Estos resultados sugieren que la optimización no lineal de hiperparámetros puede disminuir la magnitud de los parámetros entrenados, conservando niveles altos de precisión en problemas de clasificación binaria no lineal.

---

## Metodología computacional

El flujo experimental implementado en el repositorio sigue las siguientes etapas:

1. Generación de conjuntos de datos sintéticos bidimensionales.
2. Normalización de las variables de entrada.
3. Entrenamiento de una red neuronal base mediante Adam.
4. Evaluación de pérdida, precisión y norma de pesos.
5. Optimización de hiperparámetros mediante L-BFGS-B.
6. Entrenamiento de una red neuronal optimizada.
7. Comparación entre modelo base y modelo optimizado.
8. Exportación de métricas numéricas en formato JSON.
9. Generación de figuras para el análisis y el artículo.

---

## Citación

Si utiliza este repositorio, cite el software asociado al artículo:

```bibtex
@software{ortega_chavez_2026_entrenamiento_memoria_redes,
  author       = {Ortega-Chávez, Kevin L. and Fernández-Fernández, Yasmany},
  title        = {Entrenamiento eficiente en memoria de redes neuronales mediante programación no lineal},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {GitHub},
  url          = {https://github.com/USUARIO/entrenamiento-eficiente-memoria-redes-neuronales},
  doi          = {10.5281/zenodo.XXXXXXX}
}
```

Cuando el repositorio sea archivado en Zenodo, reemplace `10.5281/zenodo.XXXXXXX` por el DOI definitivo generado por la plataforma.

---

## Frase sugerida para el artículo

El código fuente, los scripts experimentales, las métricas numéricas y los archivos de reproducibilidad asociados a este estudio se encuentran disponibles en el repositorio de GitHub **“Entrenamiento eficiente en memoria de redes neuronales mediante programación no lineal”**, versión 1.0.0, archivado en Zenodo.

---

## Palabras clave

- redes neuronales
- entrenamiento eficiente en memoria
- programación no lineal
- optimización L-BFGS-B
- perceptrón multicapa
- optimizador Adam
- regularización L2
- optimización de hiperparámetros
- clasificación binaria
- reproducibilidad científica
- Python
- aprendizaje automático

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo `LICENSE` para más detalles.

---

## Autoría

**Kevin L. Ortega-Chávez**  
Universidad Politécnica Estatal del Carchi, Ecuador  
ORCID: 0009-0005-7811-6147

**Yasmany Fernández-Fernández**  
Universidad Politécnica Estatal del Carchi, Ecuador  
ORCID: 0000-0002-9530-4028

---

## Estado del repositorio

Versión inicial: `v1.0.0`  

Este repositorio forma parte de los materiales de soporte computacional y reproducibilidad del artículo científico asociado.
