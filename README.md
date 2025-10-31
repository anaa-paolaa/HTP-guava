Workflow para análisis de imágenes de guayaba por medio de PlantCV.

Para utilizar este workflow es necesario tener Python al menos en la versión 3 instalado, al igual que importado anaconda/ plantcv.**

Este análisis comienza con la descarga de imágenes al escritorio. Las imágenes se obtienen directamente de las carpetas guardadas en GoogleDrive.

| Platform  | Script Type  | Filename                 | Run Command               |
| --------- | ------------ | ------------------------ | ------------------------- |
| Windows   | Batch file   | `run_guava_wds.bat`      | `cmd`                     |
| Mac/Linux | Shell script | `run_guava_mac.sh`       | `./run_guava_pipeline.sh` |

2. Instalar los paquetes necesarios para acceder a Google con el siguiente comando y confirmar correcta instalación (no output) en la Terminal :
pip install -U pip google-api-python-client oauth2client
python3 -c "import googleapiclient, httplib2, oauth2client" #Python3 - solo en anaconda

3. Descargar las imágenes por medio del siguiente comando, modificar path si es necesario dependiendo de env local, indicando también el nombre del folder como se encuentra en Fotos_Scanner Drive. REVISAR** 29 NOV
python '/Users/anapaola/Desktop/Guava/Scripts/download_images.py'
