from src.Flight_ML.logger import logging
from src.Flight_ML.exception import CustomException
from src.Flight_ML.components.data_ingestion import DataIngestion
from src.Flight_ML.components.data_ingestion import DataIngestionConfig
from src.Flight_ML.components.model_tranier import ModelTrainerConfig,ModelTrainer
from src.Flight_ML.components.data_transformation import DataTransformationConfig,DataTransformation
import sys



if __name__=="__main__":
    logging.info("The Execution  has Started Here")


    try:
       # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
       
       # data transformation  
        data_transformation= DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transfromation(train_data_path,test_data_path)
       # model trainer
        model_trainer= ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr)) 


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)