import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.Flight_ML.utils import save_object
from src.Flight_ML.exception import CustomException
from src.Flight_ML.logger import logging


@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    #data = pd.read_excel('D:\Flight_project\artifacts\raw.csv')
    def feature_engineering(self, data):
        try:
            # Extracting date
            data["Journey_date"] = pd.to_datetime(data['Date_of_Journey'], format="%d/%m/%Y").dt.day
            # Extracting month
            data["Journey_month"] = pd.to_datetime(data['Date_of_Journey'], format="%d/%m/%Y").dt.month

            # Convert Date_of_Journey, Dep_Time, Arrival_Time to datetime
            data["Dep_Time"] = pd.to_datetime(data["Dep_Time"], format="%H:%M")
            data["Arrival_Time"] = pd.to_datetime(data["Arrival_Time"])

            #data["Dep_Time"] = pd.to_datetime(data["Dep_Time"], format="%Y-%m-%d %H:%M:%S")
            #data["Arrival_Time"] = pd.to_datetime(data["Arrival_Time"], format="%Y-%m-%d %H:%M:%S")

            # Extract hour and minute from the Dep_Time and Arrival_Time columns
            data["Departure_hour"] = data["Dep_Time"].dt.hour
            data["Departure_minute"] = data["Dep_Time"].dt.minute
            data["Arrival_hour"] = data["Arrival_Time"].dt.hour
            data["Arrival_minute"] = data["Arrival_Time"].dt.minute

            # Splitting and processing Duration using vectorized operations
            data["Duration"] = pd.to_timedelta(data["Duration"])
            data["duration_hour"] = data["Duration"].dt.components.hours
            data["duration_minute"] = data["Duration"].dt.components.minutes

            # Converting 'Date_of_Journey' column to datetime format
            data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], format="%d/%m/%Y")

            # Create a new column 'Weekend' to indicate whether the date falls on a weekend
            data['Weekend'] = (data['Date_of_Journey'].dt.dayofweek >= 5).astype(int)

            # Create 'Night' column
            data["Night"] = (data["Departure_hour"] >= 19).astype(int)

            #1) Here we see that some Airlines name belong to same airline company
            #2) So we see there is Data Inconsistency
            # Handling Data Inconsistency in 'Airline' and 'Destination'
            data['Airline'] = data['Airline'].str.replace("Vistara Premium economy", "Vistara")
            data['Airline'] = data['Airline'].str.replace("Jet Airways Business", "Jet Airways")
            data['Airline'] = data['Airline'].str.replace("Multiple carriers Premium economy", "Multiple carriers")

            data['Destination'] = data['Destination'].str.replace("New Delhi", "Delhi")

            # droping trujet airline
            data['Airline'] = data['Airline'].replace('Trujet',np.nan)

            # Converting 'Date_of_Journey' column to datetime format if not already
            data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])

            # Creating a new column to represent the day of the week (0 = Monday, 6 = Sunday)
            data['Day_of_Week'] = data['Date_of_Journey'].dt.dayofweek

            # Maping the day of the week to its name (Monday, Tuesday, etc.)
            day_name_map = {
                0: 'Monday',
                1: 'Tuesday',
                2: 'Wednesday',
                3: 'Thursday',
                4: 'Friday',
                5: 'Saturday',
                6: 'Sunday'
            }
            data['Day_of_Week'] = data['Day_of_Week'].map(day_name_map)

            # Now, 'Day_of_Week' column will contain the day names
            data['Day_of_Week'] = data['Day_of_Week'].replace({'Monday': '1', 'Tuesday': '2', 'Wednesday': '3',
                                                            'Thursday': '4', 'Friday': '5', 'Saturday': '6', 'Sunday': '7'})

            # Convert 'Day_of_Week' column to integer type
            data['Day_of_Week'] = data['Day_of_Week'].astype(int)

            # Ordinal Encoding for 'Total_Stops'
            stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
            data['Total_Stops'] = data['Total_Stops'].map(stops_mapping)
            # Drop rows with NaN values
            data = data.dropna()
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e,sys)

    def one_hot_encoding(self, final):
        try:
            Airline = pd.get_dummies(final[["Airline"]], drop_first=True, dtype=int)
            Source = pd.get_dummies(final[["Source"]], drop_first=True, dtype=int)
            Destination = pd.get_dummies(final[["Destination"]], drop_first=True, dtype=int)
            result_data = pd.concat([final, Airline, Source, Destination], axis=1)
            return pd.DataFrame(result_data)
        except Exception as e:
            raise CustomException(e,sys)
        
    def drop_columns(self, data):
        try:
            return data.drop(['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Additional_Info'], axis=1)
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_transformation_object(self):
        try:
            feature_eng_pipeline = Pipeline([
                ('feature_engineering', FunctionTransformer(self.feature_engineering)),
                ('one_hot_encoding', FunctionTransformer(self.one_hot_encoding)),
                ('drop_columns', FunctionTransformer(self.drop_columns)),
            ])

            return feature_eng_pipeline
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transfromation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj= self.get_data_transformation_object()

            target_column_name="Price"
            
            # dividing the train dataset to indepentent and depentent feature

            input_feature_train_df = train_df.drop(columns=["Price"],axis=1)
            target_feature_train_df = train_df[target_column_name]

            # dividing the test dataset to indepentent and depentent feature

            input_feature_test_df = test_df.drop(columns=["Price"],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            target_feature_train_df = target_feature_train_df.iloc[:8544]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("preprocessing applied on traing and test dataframe ")

            # As after excecution we got an error that both train array have different rows so adjusting the target_feature_train_df
            
            logging.info(f"columns names of X_train:{list(input_feature_train_arr)}")
            logging.info(f"columns names of X_test:{list(input_feature_test_arr)}")

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            #train_arr = np.concatenate((train_arr, np.zeros(1)), axis=1)

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(sys,e)