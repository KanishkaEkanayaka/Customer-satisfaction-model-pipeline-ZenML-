import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract course for developing strategy for model evaluation
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_preds: np.ndarray):
        """
        Calculate the score for the model
        Args:
            y_true: true labels
            y_preds: predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error
    """

    def calculate_score(self, y_true: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info(f"Calcuating MSE")
            mse = mean_squared_error(y_true=y_true, y_pred=y_preds)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE {e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation strategy that uses R2
    """
    def calculate_score(self, y_true: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info(f"Calculating R2 score")
            r2 = r2_score(y_true=y_true,y_pred=y_preds)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating r2 {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation strategy that uses RMSE
    """
    def calculate_score(self, y_true: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info(f"Calculating RMSE")
            rmse = mean_squared_error(y_true=y_true, y_pred=y_preds,squared=True)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except  Exception as e:
            logging.error(f"Error in calculating rmse {e}")
            raise e