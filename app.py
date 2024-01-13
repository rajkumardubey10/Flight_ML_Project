from src.Flight_ML.logger import logging
from src.Flight_ML.exception import CustomException
from src.Flight_ML.components.data_ingestion import DataIngestion
from src.Flight_ML.components.data_ingestion import DataIngestionConfig
import sys



if __name__=="__main__":
    logging.info("The Execution  has Started Here")


    try:
       # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)