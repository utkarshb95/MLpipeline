import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, selected_features, target, pivot_columns=None, fill_method='ffill', clip_columns=None):
        """
        Initialize the Preprocessor object.

        :param selected_features: List of selected feature column names.
        :param target: List of target column names.
        :param pivot_columns: Index, column and value to pivot the dataframe. Default is None.
        :param fill_method: Method for filling missing values either 'ffill' or 'bfill'. Default is 'ffill'.
        :param clip_columns: Dictionary of columns to clip with their respective min and max values. Default is None.
        """
        self.selection = selected_features
        self.target = target
        self.pivot_columns = pivot_columns
        self.fill_method = fill_method
        self.clip_columns = clip_columns or {}
    
    def clean_df(self, df):
        """
        Clean and preprocess the input DataFrame.

        :param df: Input DataFrame.
        :return: Preprocessed DataFrame.
        """
        # Pivot the table
        if self.pivot_columns:
            pivoted_df = df.pivot(index=self.pivot_columns['index'], columns=self.pivot_columns['columns'], values=self.pivot_columns['values']).reset_index()
        else:
            pivoted_df = df.copy()

        # Apply the specified fill method if provided
        if self.fill_method == 'ffill':
            pivoted_df = pivoted_df.ffill()
        elif self.fill_method == 'bfill':
            pivoted_df = pivoted_df.bfill()

        # Reset index and column names
        pivoted_df.index.names = [None] * len(pivoted_df.index.names)
        pivoted_df.columns.name = None

        # Filtered signals
        filtered_df = pivoted_df[self.selection + self.target]
        filtered_df = filtered_df.dropna().reset_index(drop=True)

        # Replace zeros with the last known non-zero value for all columns
        filtered_df = filtered_df.apply(lambda col: col.replace(0, method='ffill'))

        # Clip columns if range is provided
        if self.clip_columns:
            for col in self.clip_columns:
                print(col)
                filtered_df[col] = filtered_df[col].clip(self.clip_columns[col][0], self.clip_columns[col][1])

        return filtered_df
