# TFG - Detección y delimitación de hepatocarcinomas basándose en análisis de imagen mediante IA.

Este repositorio contiene el codigo de una aplicación web aplicación web para la clasificación automática de imágenes de ultrasonido digestivo, utilizando modelos de inteligencia artificial entrenados con imágenes reales clasificadas por médicos especialistas.

Además, incluye todos los scripts necesarios para el preprocesamiento de datos y el entrenamiento de los modelos, permitiendo ajustar, reentrenar o crear nuevos modelos desde cero según sea necesario.



## Tabla de Contenidos

- [Aplicación Web](#aplicación-web)
- [Entrenamiento de Modelos](#entrenamiento-de-modelos)
- [Preprocesamiento](#preprocesamiento)

## Aplicación Web 

Para lanzar la aplicación ejecute, desde la raíz del proyecto, el siguiente comando :
```bash 
docker compose up
``` 
> Para ello necesitará tener instalado [Docker](https://docs.docker.com/get-docker/) y [Docker Compose](https://docs.docker.com/compose/install/), con sulte los enlaces para realizar la instalación.  
> Dependiendo de su instalación puede que necesite darle permisos de super usuario a docker. 

Tambien puede utilizar el script `rebuild.sh` que:
1. Dentendrá posibles contenedores y volumenes montados. 
2. Borrará los que no esten siendo usados.
3. Ejecutará los contenedores forzando a construir y recrear las imagenes.
 
Es útil para reiniciar el entorno en caso de fallos o al aplicar cambios en el código.

### Modelos

Para que la aplicación pueda realizar tareas de clasificación debe de tener los modelos de inteligencia articial en sus respectivas carpetas sobre las que Docker montará volumenes. 
Puede lanzar la aplicación sin modelos, aunque no podrá clasificar con ella. Mas tarde, puede entrenar los modelos o reentrarlos sin tener que relanzar la aplicación.

Los modelos se guardan en 
```
data/model_state/modo_funcionamiento>/<tipo_modelo>/
```

## Entrenamiento de Modelos

Para poder entrenar los modelos necesitará primero instalar los paquetes de Python corresnecesariospondientes. Para ello use:
```
pip install -r requirements.txt
```

Hay dos tipos de modelos en este proyecto y cada uno tiene sus propios scripts de entrenamiento, que deben ejecutarse desde la raiz del proyecto:
 + **Modelo Personalizado**. Ejecute: 
 ```bash
 python -m code.scripts.models.train_custom
 ``` 

 + **Modelos Preentrenados**. Ejecute:
 ```bash
 python -m code.scripts.models.train_pretrained
 ``` 
Ambos scripts le indicarán los argumentos necesaros para su funcionamiento.

Si en algun momento se detiene la ejecución del Modelo Personalizado se puede retomar utilizando 
```bash
python -m code.scripts.models.retrain_custom
``` 
Utilizando los argumentos que le indica el script.

### Alternativa con PYTHONPATH

Si lo prefiere puede exportar la raíz del proyecto como PYTHONPATH 
```bash
export PYTHONPATH=$(pwd)
```
Esto permitirá ejecutar los archivos directamente, aunque necesitará:
+ Darles permisos de ejecución:
```bash
chmod +x ruta_al_archivo.py
``` 
+ o utilzar el interprete de python 
```bash
python ruta_al_archivo.py
```
### Requisitos de datos
Para entrenar los modelos necesitará tener imagenes dentro de las carpetas de 
```bash
data/images/HURH/<categoria>/<modo_ecografia>/
data/images/OneDrive/<categoria>/1280x960/<modo_ecografia>/
``` 
y sus correspondientes archivos CSV en 
```bash
data/csv/final/
``` 
Si no los tiene tendrá que preprocesar las imagenes y generar los csv.

## Preprocesamiento

Hay dos fuentes de imagenes:

  + Las recogidas en mano del ecografo "Aplio i700" del HURH (Hospital Univesitario Rio Hortega). Carpeta: 
  ```bash
  data/images/HURH
  ```
   Dentro hay 8 categorias de imagenes.

  + Las obtenidas a traves del OneDrive compartido por los medicos. Carpeta: 
  ```bash
  data/images/OneDrive
  ```
   Dentro hay 3 categorias de imagenes. 


Las imagenes de ambas fuentes han sido clasificadas a mano por medicos especializados en el sistema digestivo.   
Ambas se usan para entrenar los modelos, pero no es necesario que ambas tengan imagenes para hacerlo.

### Ejecución del preprocesamiento

Para realizar el preparar los datos ejecute: 
```bash
python -m code/scripts/preprocessing/pipeline_classification
```

Este script se encargará de: 
1. Eliminar artefactos y anotaciones de la interfaz de las imagenes de ultrasonidos.
2. Separar las imagenes en las que usan la tecnica Dópler de las que no. Carpetas 'bmode' y 'doppler'.
3. En el caso de OneDrive encontar las imagenes compatibles con las del "Applio i700" y realizar los pasos 1 y 2.

Además, se generan archivos CSV con anotaciones para: 
+ Registrar  las rutas a las imagenes.
+ Indicar la tecnica utilizada(Modo B o Dópler).
+ Posibiilta la reconstrucción del proceso en caso de rror.
