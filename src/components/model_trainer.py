import os 
import sys 
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import dill
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_path:str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing data")

            X_train,Y_train,X_test,Y_test = (
                train_array[:,:-1], # take all rows but keep columns except last one 
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            ) 
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            models_report:dict = evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)

            best_model_score = max(sorted(models_report.values()))

            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")

            logging.info("Best Model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
