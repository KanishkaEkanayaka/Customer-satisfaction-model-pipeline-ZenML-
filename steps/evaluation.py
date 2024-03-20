import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"]
]:
    """
    Evaluates the model
    Args:
        model: model to evaluate
        X_test: testing data
        y_test: testing labels
    Returns:
        (r2,rmse) : r2 score and root mean squared error
    """

    try:
        prediction = model.predict(X_test)
        
        mse_class = MSE()
        mse = mse_class.calculate_score(y_true=y_test,y_preds=prediction)
        mlflow.log_metric("mse",mse)

        r2_class = R2()
        r2 = r2_class.calculate_score(y_true=y_test, y_preds= prediction)
        mlflow.log_metric("r2",r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_true=y_test, y_preds=prediction)
        mlflow.log_metric("rmse",rmse)

        return r2, rmse
    except Exception as e:
        logging.error("Error while evaluating model {e}")
        raise e