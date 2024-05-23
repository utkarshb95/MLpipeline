import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

class Trainer:
    def __init__(self, model, use_standardscaler=False, session_id='drive_no'):
        """
        Initialize the SklearnTrainer.

        Args:
            model: An instance of the scikit-learn model.
            use_standard_scaler (bool): Whether to use StandardScaler preprocessing (default: False).
        """
        self.model = model
        self.use_standardscaler = use_standardscaler
        self.session_id = session_id
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        steps = []
        if self.use_standardscaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', self.model))
        return Pipeline(steps)
    
    def train(self, df, features, target, test_exclude=0, verbose=0):
        """
        Train the scikit-learn model.

        Args:
            df (pd.DataFrame): The input dataframe.
            features (list): List of input feature columns.
            target (str): Target column name.
            test_exclude (int): Drive number to exclude from training (default: 0).
            verbose (int): Verbosity level (0: no output, 1: basic output).

        Returns:
            trained_pipeline, mse, r2, signature
        """
        if test_exclude != 0:
            train_df = df[~df[self.session_id].isin([int(test_exclude)])]
            if verbose:
                print("Training on drives: ", set(train_df[self.session_id]))
        else:
            train_df = df
            if verbose:
                print("Training on drives: ", set(train_df[self.session_id]))

        # Exclude the specified column if it exists using isin
        input_columns = train_df[features].loc[:, ~train_df[features].columns.isin([self.session_id])]
        
        signature = infer_signature(model_input=input_columns)
        trained_pipeline = self.pipeline.fit(input_columns, train_df[target])

        # Evaluate model performance
        if verbose:
            train_predictions = trained_pipeline.predict(input_columns)
            mse = mean_squared_error(train_df[target], train_predictions)
            r2 = r2_score(train_df[target], train_predictions)
            print("Mean Squared Error:", mse)
            print("R-squared:", r2)

        return trained_pipeline, mse, r2, signature