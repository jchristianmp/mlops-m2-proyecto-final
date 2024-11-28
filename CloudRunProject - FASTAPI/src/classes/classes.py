
import joblib 
import numpy as np
import pandas as pd
import os

from feature_engine.encoding import OrdinalEncoder
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

import Input.preprocessors as pp
from configuraciones import config



class classPrediction:
    def __init__(self) -> None:
        pass

    def prediccion_o_inferencia(self, datos_de_test, ruta_actual): 
        
        #Cargar el Modelo ML o Cargar el Pipeline

        ruta_modelo = os.path.join(ruta_actual, "src/precio_coches_pipeline.joblib")
        pipeline_de_produccion = joblib.load(ruta_modelo)

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

    def contac_prediccion(self, datos_procesados, prediccion, prediccion_sin_escalar, ruta_actual):
        
        # Concatenar los datos procesados y las predicciones
        #df_concatenado = pd.concat([datos_procesados, pd.Series(prediccion, name="Predicciones"), pd.Series(prediccion_sin_escalar, name="Predicciones_Sin_Escalar")], axis=1)

        # Guardar el archivo de salida
        #output_file = f"{ruta_actual}/salida_datos_y_predicciones.csv"
        #df_concatenado.to_csv(output_file, index=False)

        #df_resultado = datos_procesados.copy()
        #df_resultado['Predicción Escalada'] = prediccion
        #df_resultado['Predicción Sin Escalar'] = prediccion_sin_escalar

        # Concatenar los datos procesados y las predicciones
        df_concatenado = pd.concat([datos_procesados, 
                                    pd.Series(prediccion, name="Predicciones"), 
                                    pd.Series(prediccion_sin_escalar, name="Predicciones_Sin_Escalar")], 
                                    axis=1)

        # Guardar el archivo de salida
        output_file = f"{ruta_actual}/salida_datos_y_predicciones.csv"
        df_concatenado.to_csv(output_file, index=False)
        
        return df_concatenado.to_csv(output_file, index=False)