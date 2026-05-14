Este repositorio contiene la implementacion, vectorizacion y el analisis estadistico de tres escenarios practicos orientados a la ingenieria de software y datos, utilizando tecnicas de simulacion estocastica y muestreo avanzado.

El proyecto esta estructurado siguiendo buenas practicas de ingenieria, separando la logica algoritmica en una libreria modular de Python y la visualizacion en un entorno de Jupyter Notebook.

Estructura del Proyecto
El repositorio se organiza de la siguiente manera:

src/simulacion_mc/: Paquete principal de la libreria.

init.py: Archivo de inicializacion del modulo.

rag_latency.py: Logica de simulacion para latencias RAG utilizando distribuciones Normal, Uniforme y Lognormal.

bio_is.py: Implementacion de Importance Sampling para estimacion de eventos raros.

bimodal_rs.py: Algoritmo de Rejection Sampling para generacion de trafico bimodal.

Evaluacion_Simulacion.ipynb: Notebook principal con la ejecucion, graficas y analisis de resultados.

pyproject.toml: Configuracion de empaquetado y dependencias del proyecto.

venv/: Entorno virtual de Python (excluido en el control de versiones).

Instalacion y Configuracion
Siga estos pasos para configurar el entorno local:

Clonar el repositorio:
git clone https://github.com/Jhorman-13/Simulacion-Montecarlo.git
cd Taller_Tres_Modelos_Simulacion

Crear y activar el entorno virtual:
python -m venv venv
source venv/Scripts/activate

Instalar dependencias y el paquete en modo editable:
pip install -e .
pip install notebook matplotlib numpy scipy ipykernel

Configurar el Kernel en VS Code:
Abra el archivo .ipynb y seleccione el interprete de Python ubicado en la carpeta venv.

Analisis de Casos de Estudio
Caso 1: Analisis de Latencia en Arquitecturas RAG
Se simularon 100,000 consultas para evaluar el cumplimiento de un SLA de 300 ms.

Resultados Obtenidos: El Percentil 95 se situo en 276.09 ms, mientras que el Percentil 99 alcanzo los 358.70 ms.

Conclusion: El sistema no cumple con el SLA para el percentil 99. La distribucion muestra una cola larga hacia la derecha causada por la variabilidad de la etapa de inferencia del LLM, la cual requiere optimizacion para estabilizar los tiempos de respuesta.

Caso 2: Cuellos de Botella en Pipelines (Importance Sampling)
Se busco estimar la probabilidad de fallo en ejecuciones que superan las 150 horas.

Resultados Obtenidos: Se calculo una probabilidad de fallo aproximada del 2.05% al 2.25%.

Conclusion: La tecnica de Importance Sampling permitio una reduccion de varianza de 1.08x respecto al metodo estandar. Aunque la mejora es modesta, el metodo demuestra ser eficaz para enfocar la simulacion en regiones criticas de fallo.

Caso 3: Trafico Bimodal (Rejection Sampling)
Se genero una distribucion de trafico sintetico con dos picos de actividad.

Resultados Obtenidos: Se obtuvo una tasa de aceptacion del 16.67%.

Conclusion: El algoritmo logro replicar con precision la forma de la distribucion bimodal objetivo. No obstante, la baja tasa de aceptacion indica una ineficiencia computacional debido al uso de una envolvente uniforme excesivamente amplia, sugiriendo el uso de mixturas de gaussianas para futuras optimizaciones.

Tecnologias Utilizadas
Lenguaje: Python 3.14

Analisis de Datos: NumPy, SciPy

Visualizacion: Matplotlib

Documentacion: Markdown

Entorno: Jupyter Notebook / VS Code
