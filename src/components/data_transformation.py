import os 
import sys 

from dataclasses import dataclass 

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.logger import logging 
from src.exception import CustomException
from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    preprocessor_ob_file_path:str = os.path.join('artifacts',"preprocessor.pkl") 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self): # for creating pkl file 
        '''
        this function is responsible for data transformation
        '''
        try:

            num_columns = ["reading_score","writing_score"]
            cat_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            # handles missing values,converting into numerical values
            cat_pipeline = Pipeline( 
                steps = [
                    ("cat_imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical Columns Scaling Completed")
            logging.info("Categorical Columns Encoding Completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_columns),
                    ("cat_pipeline",cat_pipeline,cat_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the train and test data")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_transformer_object()

            target_col_name = "math_score"

            input_features_train_df = train_df.drop(columns=[target_col_name],axis=1)
            target_features_train_df = train_df[target_col_name]

            input_features_test_df = test_df.drop(columns=[target_col_name],axis=1)
            target_features_test_df = test_df[target_col_name]

            logging.info("Applying preprocessing object on trianing and testing dataframe.")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_arr,np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_arr,np.array(target_features_test_df)]

            logging.info("Preprocessing Done")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e :
            raise CustomException(e,sys)