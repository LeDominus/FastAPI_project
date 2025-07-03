import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

DATA_PATH = os.getenv('DATA_PATH', 'models/data/AAPL.csv')

class ModelTrainer():
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)
        self.scaler = StandardScaler()
        self.model = LinearRegression(n_jobs=-1)
    
    def preprocess_data(self):
        if self.df['Close'].isna().sum() > 0:
            self.df['Close'] = self.df['Close'].fillna(0.01)
            
        df_new = self.df.drop(['Date'], axis=1)
        
        df_scaled = self.scaler.fit_transform(df_new)
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df_new), columns=columns)
        df_scaled['Date'] = self.df['Date']

        X = df_scaled.drop(['Close', 'Date'], axis=1)
        y = df_scaled['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        joblib.dump(self.model, 'models/artifacts/model.joblib')
        joblib.dump(self.scaler, 'models/artifacts/scaler.joblib')
        joblib.dump(X_test, 'models/artifacts/X_test.joblib')
        joblib.dump(y_test, 'models/artifacts/y_test.joblib')
        
trainer = ModelTrainer()