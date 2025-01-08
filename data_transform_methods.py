"""Methods for transforming data types in a pandas Dataframe."""

import pandas as pd


class DataTransform:
    """
    Methods for transforming data type in a pandas Dataframe.
    """
    @staticmethod
    def transform_to_category(df, column):
        """
        Converts a dataframe column to category data type
        
        Args:
            df (pandas.Dataframe): A pandas Dataframe.
            column (str): A column in the pandas Dataframe, df.

        Returns:
            df[column] (pandas.Series): the column following the data type conversion. 
        """
        df[column] = df[column].astype('category')
        return df[column]
    
    @staticmethod
    def transform_to_boolean(df, column):
        """
        Converts a dataframe column to boolean data type
        
        Args:
            df (pandas.Dataframe): A pandas Dataframe.
            column (str): A column in the pandas Dataframe df.

        Returns:
            df[column] (pandas.Series): the column following the data type conversion.
        """
        df[column] = df[column].map({'y': True, 'n': False})
        return df[column]
    
    @staticmethod
    def transform_to_datetime(df, column):
        """
        Converts a dataframe column to datetime data type
        
       Args:
            df (pandas.Dataframe): a pandas Dataframe.
            column (str): A column in the pandas Dataframe df.

        Returns:
            df[column] (pandas.Series): the column following the data type conversion. 
        """
        df[column] = pd.to_datetime(df[column], errors='coerce', format='mixed')
        return df[column]

    @staticmethod
    def transform_to_numeric(df, column):
        """
        Converts a dataframe column to numeric data type
        
       Args:
            df (pandas.Dataframe): a pandas Dataframe.
            column (str): A column in the pandas Dataframe df.

        Returns:
            df[column] (pandas.Series): the column following the data type conversion. 
        """
        df[column] = pd.to_numeric(df[column], errors='coerce')
        return df[column]