import os 
import sys
from dataclasses import dataclass
from src.utils import evaluate

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.expection import CustomExpection 
from src.logger import logging
from src.utils import save_object


@dataclass 
class ModelTrainConfig:
    trained_model_data_path = os.path.join("artifacts","model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()

    def intiate_model_training(self,train_array,test_array):
        try:
            logging.info("train and split the data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            model={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
            }
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "absolute_error",
                        "poisson"
                    ]
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "KNeighbors Regressor": {},

                "Linear Regression": {},

                "XGB Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },

                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            model_report:dict=evaluate(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=model,param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model.keys())[
                list(model_report.values()).index(best_model_score)
            ]
 
            best_model = model[best_model_name]

            if best_model_score<0.6:
                raise CustomExpection("No best model found")
            logging.info("Best Model Found on training and testing data path")

            save_object(
                file_path=self.model_trainer_config.trained_model_data_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_scor = r2_score(y_test,predicted)

            return r2_scor
            
        except  Exception as e:
            raise CustomExpection(e,sys)
    

