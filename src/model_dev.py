import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
import pandas as pd

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model
        Args:
            X_train: Training data
            y_train: training labels
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear regression model
    """

    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        """
        Train the model
        Args:
            X_train: Training data
            y_train: training labels
        Returns:
            reg: RegressionModel
        """

        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info(f"Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error in model training {e}")
            raise e
        
# IF you want more models to train create classes for each model to develop here.