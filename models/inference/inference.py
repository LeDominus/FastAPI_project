import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../artifacts/model.joblib')
X_TEST_PATH = os.path.join(BASE_DIR, '../artifacts/X_test.joblib')
Y_TEST_PATH = os.path.join(BASE_DIR, '../artifacts/y_test.joblib')
SCALER = os.path.join(BASE_DIR, '../artifacts/scaler.joblib')

model = joblib.load(MODEL_PATH)
X_test = joblib.load(X_TEST_PATH)
y_test = joblib.load(Y_TEST_PATH)
scaler = joblib.load(SCALER)

class ModelPredictor():
    def __init__(self):
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.scaler = scaler
        self.prediction = None 
    
    def predict(self):
        self.prediction = self.model.predict(self.X_test)
        return self.prediction
    
    def metrics(self) -> dict:
        
        if self.prediction is None:
            self.predict()
        
        r2 = r2_score(self.y_test, self.prediction) * 100
        mape = mean_absolute_percentage_error(self.y_test, self.prediction) * 100
        rmse = root_mean_squared_error(self.y_test, self.prediction)
        return {'R2': np.round(r2, 4), 'MAPE': np.round(mape, 4), 'RMSE': np.round(rmse, 4)}

predictor = ModelPredictor()


