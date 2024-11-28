
import joblib 
import numpy as np

from feature_engine.encoding import OrdinalEncoder
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

import Input.preprocessors as pp
from configuraciones import config



class classPrediction:
    def __init__(self) -> None:
        pass

    def prediccion_o_inferencia(self, datos_de_test): 
        
        #Cargar el Modelo ML o Cargar el Pipeline
        pipeline_de_produccion = joblib.load('src/precio_coches_pipeline.joblib')

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

    def contac_prediccion(self, datos_procesados, prediccion, prediccion_sin_escalar):
        df_resultado = datos_procesados.copy()
        df_resultado['Predicción Escalada'] = prediccion
        df_resultado['Predicción Sin Escalar'] = prediccion_sin_escalar
        
        # Creamos el archivo CSV para descargar
        csv = df_resultado.to_csv(index=False).encode('utf-8')

        return df_resultado, csv