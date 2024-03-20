import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaner, DataDivideStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Cleans the data and divide it.

    Args:
        df: Raw data
    Returns:
        X_train: training data
        X_test: testing data
        y_train: traininf labels
        y_test: testing labels
    """
    try:
        data_process = DataCleaner(data=df,strategy=DataPreProcessStrategy)
        data = data_process.handle_data()
        data_split = DataCleaner(data=data,strategy=DataDivideStrategy)
        X_train, X_test, y_train, y_test = data_split.handle_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error while cleaning data {e}")
        raise e