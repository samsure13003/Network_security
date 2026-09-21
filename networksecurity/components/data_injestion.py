from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


## configuration of the Data Ingestion Config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

# Target column used for stratified splitting, matching the EDA/training notebook.
# Falls back to "Result" (the raw label column, -1 = phishing / 1 = legitimate) if the
# constant isn't defined elsewhere in the project.
try:
    from networksecurity.constant.training_pipeline import TARGET_COLUMN
except ImportError:
    TARGET_COLUMN = "Result"


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from mongodb
        """
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"])
            
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e

    def log_data_quality_summary(self, dataframe: pd.DataFrame) -> None:
        """
        Log missing-value and duplicate-row counts for this run, purely for
        visibility in the log folder. Does not modify the dataframe.
        """
        try:
            missing_counts = dataframe.isnull().sum()
            total_missing = int(missing_counts.sum())
            total_duplicates = int(dataframe.duplicated().sum())

            logging.info(
                f"Data quality summary -- rows: {len(dataframe)}, "
                f"total missing values: {total_missing}, "
                f"duplicate rows: {total_duplicates}"
            )
            if total_missing > 0:
                logging.info(
                    f"Missing values by column:\n"
                    f"{missing_counts[missing_counts > 0].to_string()}"
                )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def remove_duplicate_rows(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Drop exact duplicate rows before splitting.

        The training notebook's experiment showed the raw data is ~47% exact
        duplicates, and leaving them in causes train/test leakage (64.6% of test
        rows were memorized copies of training rows), inflating reported accuracy.
        De-duplicating here, before the train/test split, avoids that leakage.
        """
        try:
            rows_before = len(dataframe)
            dataframe = dataframe.drop_duplicates().reset_index(drop=True)
            rows_after = len(dataframe)
            logging.info(
                f"Removed {rows_before - rows_after} duplicate rows "
                f"({rows_before} -> {rows_after})"
            )
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            stratify_col = (
                dataframe[TARGET_COLUMN] if TARGET_COLUMN in dataframe.columns else None
            )
            if stratify_col is None:
                logging.info(
                    f"Target column '{TARGET_COLUMN}' not found; "
                    f"splitting without stratification"
                )

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                stratify=stratify_col,
                random_state=42,
            )
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            self.log_data_quality_summary(dataframe)
            dataframe=self.remove_duplicate_rows(dataframe)
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e