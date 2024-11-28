from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse

import pandas as pd 
import numpy as np
import joblib
import os

from feature_engine.encoding import OrdinalEncoder
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

import Input.preprocessors as pp
from configuraciones import config


# Instancia FastAPI
app = FastAPI()

ruta_actual = os.getcwd()

def prediccion_o_inferencia(pipeline_de_produccion, datos_de_test): 
        
        #Cargar el Modelo ML o Cargar el Pipeline
        # Quitamos etiquetas km y miles y convertimos a numerico
        datos_de_test["running"] = datos_de_test["running"].astype(str)
        datos_de_test["running"] = datos_de_test["running"].str.replace(" km","")
        datos_de_test["running"] = datos_de_test["running"].str.replace(" miles","")
        datos_de_test["running"] = datos_de_test["running"].astype(float)
        datos_de_test["running"]

        datos_de_test = datos_de_test[config.FEATURES]

        # Predicciones con pipeline generado
        predicciones = pipeline_de_produccion.predict(datos_de_test)

        # Revirtiendo escalamiento
        predicciones_sin_escalar = np.exp(predicciones)
        return predicciones, predicciones_sin_escalar, datos_de_test 

@app.post("/predecir_precios")
def publicar(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
    
    try:
        # Guardar el archivo CSV subido temporalmente
        file_location = f"{ruta_actual}/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        # Leer el archivo CSV
        df_de_los_datos_subidos = pd.read_csv(file_location)

        # Cargar el pipeline de producción desde la ruta correcta
        ruta_modelo = os.path.join(ruta_actual, "src/precio_coches_pipeline.joblib")  # Actualización de la ruta
        pipeline_de_produccion = joblib.load(ruta_modelo)

        # Hacer la predicción
        predicciones, predicciones_sin_escalar, datos_procesados = prediccion_o_inferencia(pipeline_de_produccion,
                                                                                           df_de_los_datos_subidos)
        
       
        # Concatenar los datos procesados y las predicciones
        df_concatenado = pd.concat([datos_procesados, 
                                    pd.Series(predicciones, name="Predicciones"), 
                                    pd.Series(predicciones_sin_escalar, name="Predicciones_Sin_Escalar")], 
                                    axis=1)

        # Guardar el archivo de salida
        output_file = f"{ruta_actual}/salida_datos_y_predicciones.csv"
        df_concatenado.to_csv(output_file, index=False)

        # Devolver el archivo resultante
        return FileResponse(output_file, media_type="application/octet-stream", filename="salida_datos_y_predicciones.csv")

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío o tiene un formato incorrecto.")
    #except joblib.externals.loky.process_executor.TerminatedWorkerError:
    #    raise HTTPException(status_code=500, detail="Error al cargar el modelo de predicción.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")