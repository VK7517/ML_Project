import os 
import sys 
from src.exception import CustomException
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer 

@dataclass
class DataIngestionConfig: 
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestionconfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the data as dataframe")

            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path),exist_ok = True)

            df.to_csv(self.ingestionconfig.raw_data_path, index=False , header=True)

            logging.info("Train Test split Initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestionconfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")
            return {
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            }
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_array,test_array,_=data_transformation_obj.initiate_data_transformation(train_path,test_path)

    model_trainer_obj = ModelTrainer()
    print(model_trainer_obj.initiate_model_trainer(train_array,test_array))