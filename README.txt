Early Stopping para Entrenamiento de Redes de Clasificación Basado en Estimadores de Información Mutua
Autores: Diego Cabezas, Martín Moreno, Francisco Soto
-2023

Intrucciones de uso:
- El codigo fue desarrollado en Python 3.9
- Se recomienda usar un entorno virtual para la ejecución del código, el comando al utilizar conda es el siguiente
    '''conda create -n 'env-name' python=3.9'''
-Para instalar las dependencias debe ejecutar el siguiente comando:
    '''pip install -r requirements.txt'''
- Para ejecutar el código debe posicionarse en la carpeta del proyecto y ejecutar el siguiente comando:
   ''' bash experiments_scripts/experiment.sh'''
-Para editar los parametros del experimento se debe editar el archivo ubicado en 'experiments_scripts':
        experiment.sh
- El script anterior ejecutará el experimento base que se encuentra en el archivo src/base_experiment.py

En la carpeta binary_experiment se encuentra el codigo para el experimento binario desarrollado en la primera entrega.
En la carpeta hd_experiment se encuentra el codigo para el experimento del dataset heat disease desarrollado en la segunda entrega.

