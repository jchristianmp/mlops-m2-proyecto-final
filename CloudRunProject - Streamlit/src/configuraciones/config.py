import datetime

# Variables temporales
TEMPORAL_VARS = ['year']
REF_VAR = datetime.datetime.now().year


# Variables para tranformaciion yeo 
NUMERICALS_YEO_VARS = ['running', 'motor_volume', 'year'] # Lasso
#NUMERICALS_YEO_VARS = ['year'] # Linear Regression

# Variables categoricas para codificar
CATEGORICAL_VARS = ['model', 'motor_type', 'color', 'type', 'status']  # Lasso
#CATEGORICAL_VARS = ['model']  # Linear Regression

# Variables seleccionadas del proceso feature selection
FEATURES = [
            'model',
            'year',
            'motor_type',
            'running',
            'color',
            'type',
            'status',
            'motor_volume',
]