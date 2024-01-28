import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.Flight_ML.exception import CustomException
from src.Flight_ML.logger import logging

from src.Flight_ML.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            #num_columns_X_train = X_train.shape[1]
            #num_columns_X_test = X_test.shape[1]

            #name_columns_X_train = X_train.columns
            #name_columns_X_test= X_test.columns
            #logging.info("Splited training and test input data successfully")
            #logging.info("columns count of X_train: %s",num_columns_X_train)
            #logging.info("columns count of X_test: %s",num_columns_X_test)
            # columns 
            #logging.info("columns names of X_train:",num_columns_X_train)
            #logging.info("columns names of X_test:",num_columns_X_test)

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                
                "Decision Tree": {'criterion':['friedman_mse'] ,
                    'max_depth': [30],
                    'max_features': ['sqrt'],
                    'min_samples_leaf': [4],
                    'min_samples_split': [10]},

                "Random Forest":{'n_estimators': [10],
                    'min_samples_split': [10],
                    'min_samples_leaf': [4],
                    'max_features': ['sqrt'],
                    'max_depth':[ None],
                    'criterion': ['poisson'],
                    'bootstrap': [False]},

                "Gradient Boosting":{
                     'loss':['huber'],
                    'learning_rate':[0.2],
                    'subsample':[0.9],
                     'criterion':['squared_error', 'friedman_mse'],
                     'max_features':['log2'],
                    'n_estimators': [200]
                },
                
                "Linear Regression":{},
                
                "XGBRegressor":{'subsample': [0.8],
                    'n_estimators': [200],
                    'min_child_weight':[ 1],
                    'max_depth': [5],
                    'learning_rate':[ 0.2],
                    'gamma': [0.1],
                    'colsample_bytree': [0.8]
                    },
                
                "CatBoosting Regressor":{
                    'depth': [10],
                    'learning_rate': [0.05],
                    'l2_leaf_reg':[ 5],
                    'iterations': [200]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info("Model training started")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,params=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            #if best_model_score<0.6:
            #    raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)
