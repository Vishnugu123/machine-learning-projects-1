import os 
import sys 
import numpy as np
import pandas as pd
import dill 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.expection import CustomExpection

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomExpection

def evaluate(x_train, y_train, x_test, y_test, models, param):
    try:
        reports = {}

        for model_name, model in models.items():

            para = param[model_name]

            # Create GridSearchCV
            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3
            )

            # Find best parameters
            gs.fit(x_train, y_train)

            # Set best parameters to actual model
            model.set_params(**gs.best_params_)

            # Train model with best parameters
            model.fit(x_train, y_train)

            # Predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Scores
            train_model_score = r2_score(
                y_train,
                y_train_pred
            )

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            reports[model_name] = test_model_score

        return reports

    except Exception as e:
        raise CustomExpection(e, sys)

def load_object(file_path):
    try:
        with open (file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomExpection(e,sys)