"""
Methods for performing descriptive statistics and data analysis on a pandas DataFrame.

This module provides the `DataFrameInfo` class which contains static methods for performing
descriptive statistics and exploratory data analysis (EDA) on a pandas DataFrame. 
The methods allow users to get general information, summary statistics, and specific calculations like 
mean, median, mode, standard deviation, skewness, interquartile range, z-scores, and more.

Typical usage example:

    info_methods = DataFrameInfo()
    df_info = info_methods.df_info()
"""

import numpy as np


class DataFrameInfo:
    """
    Static methods for performing various descriptive statistics and data analysis. 
    
    This class provides static methods for performing various descriptive statistics
    and exploratory data analysis (EDA) operations on pandas DataFrames. 
    Including methods for calculating summary statistics, handling missing values,
    identifying outliers, and performing other useful analysis to understand the distribution
    and properties of data.
    """

    @staticmethod
    def df_info(df):
        """
        Display basic information about the DataFrame.
         
        Specifically the column names, non-null values and data types.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed. 

        Returns:
            None: Prints information about the DataFrame.
        """
        return print(df.info())
    
    @staticmethod
    def describe_df(df):
        """
        Display summary statistics for each numeric column in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.

        Returns:
            None: Prints descriptive statistics for the DataFrame.
        """
        return print(df.describe())
    
    @staticmethod
    def describe_col(df, column):
        """
        Display summary statistics for a specific column in the Dataframe.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.
            column (str): The name of the column to be described.

        Returns:
            None: Prints descriptive statistics for the specified column.
        """
        return print(df[column].describe())
    
    @staticmethod
    def df_shape(df):
        """
        Return the shape (rows, columns) of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.

        Returns:
            tuple: A tuple representing the shape (rows, columns) of the DataFrame.
        """
        return print(df.shape)
    
    @staticmethod
    def num_unique(df):
        """
        Return the number of unique values in each column of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.

        Returns:
            None: Prints the number of unique values for each column.
        """
        return print(df.nunique())
    
    @staticmethod
    def count_nulls(df):
        """
        Return the proportion of missing values in each column of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.

        Returns:
            None: Prints the proportion of missing values for each column.
        """
        return print(df.isnull().sum()/len(df))

    @staticmethod
    def df_mean(df):
        """
        Return the mean of all numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.

        Returns:
            pd.Series: A Series containing the mean of each numeric column.
        """
        return print(df.mean(numeric_only=True))
   
    @staticmethod
    def column_mean(df, column):
        """
        Return the mean of a specific column.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.
            column (str): The name of the column for which the mean is to be calculated.

        Returns:
            float: The mean of the specified column.
        """
        return df[column].mean()
    
    @staticmethod
    def df_median(df):
        """
        Return the median of all numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.

        Returns:
            pd.Series: A Series containing the median of each numeric column.
        """
        return print(df.median(numeric_only=True))
   
    @staticmethod
    def column_median(df, column):
        """
        Return the median of a specific column.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.
            column (str): The name of the column for which the median is to be calculated.

        Returns:
            float: The median of the specified column.
        """
        return df[column].median()
    
    @staticmethod
    def column_mode(df, column):
        """
        Return the mode (most frequent value) of a specific column.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.
            column (str): The name of the column for which the mode is to be calculated.

        Returns:
            pd.Series: The mode of the specified column.
        """
        return df[column].mode()

    @staticmethod
    def df_std_dev(df):
        """
        Calculates and prints the standard deviation of all numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.

        Returns:
            None: Prints the standard deviation for each numeric column.
        """
        return print(df.std(numeric_only=True))
   
    @staticmethod
    def column_std_dev(df, column):
        """
        Calculates and prints the standard deviation of a specific column in the DataFrame.

        Args:
            df (pd.DataFrame):  The DataFrame to be analysed.
            column (str): The column name for which the standard deviation is calculated.

        Returns:
            None: Prints the standard deviation of the specified column.
        """
        return print(df[column].std())
    
    @staticmethod
    def count_distinct_values(df, column):
        """
        Counts and prints the number of distinct values in a specified column of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.
            column (str): The column name for which distinct values are counted.

        Returns:
            None: Prints the distinct values and their counts in the specified column.
        """
        return print(df[column].value_counts())
    
    @staticmethod
    def df_skew(df):
        """
        Calculates and prints the skewness of all numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be analysed.

        Returns:
            None: Prints the skewness for each numeric column.
        """
        return (print(df.skew(numeric_only=True)))
    
    @staticmethod
    def col_skew(df,column):
        """
        Calculates and prints the skewness of a specific column in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the column.
            column (str): The column name for which the skewness is calculated.

        Returns:
            None: Prints the skewness of the specified column.
        """
        return (print(df[column].skew()))
    
    @staticmethod
    def measure_iqr(df, column):
        """
        Calculates the Interquartile Range (IQR) and outliers for a column in a DataFrame.

        Calculates and prints the Interquartile Range (IQR) for a specific column in the DataFrame
        and identifies outliers based on the IQR method.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the column.
            column (str): The column name for which the IQR and outliers are calculated.

        Returns:
            pd.Series: A boolean series indicating the outliers in the specified column.
        """
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        print(f"Q1 (25th percentile): {q1}")
        print(f"Q3 (75th percentile): {q3}")
        print(f"IQR: {iqr}")
        outliers = (df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))
        return outliers
    
    @staticmethod
    def calc_z_score(df, column):
        """
        Calculates the Z-score for a specific column in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the column.
            column (str): The column name for which the Z-scores are calculated.

        Returns:
            pd.Series: A series containing the Z-scores for the specified column.
        """
        mean_col = np.mean(df[column])
        std_col = np.std(df[column])
        z_scores = (df[column] - mean_col) / std_col
        return z_scores

    @staticmethod
    def lower_bound(df, column):
        """
        Calculates the lower bound for detecting outliers in a specific column using the IQR method.

        Args:
            df (pd.DataFrame): The DataFrame containing the column.
            column (str): The column name for which the lower bound is calculated.

        Returns:
            float: The lower bound value for the column to detect outliers.
        """
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr

    @staticmethod
    def upper_bound(df, column):
        """
        Calculates the upper bound for detecting outliers in a specific column using the IQR method.

        Args:
            df (pd.DataFrame): The DataFrame containing the column.
            column (str): The column name for which the upper bound is calculated.

        Returns:
            float: The upper bound value for the column to detect outliers.
        """
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        return q3 + 1.5 * iqr