"""
Data transformation methods for pandas Dataframes.

This module provides a set of transformation methods for modifying data within pandas DataFrames. 
It includes functions for common data transformations such as:

- Dropping specific columns or rows with null values
- Performing logarithmic and statistical transformations (Box-Cox, Yeo-Johnson)
- Calculating the loan term based on given financial parameters
- Removing outliers based on column values

These methods are designed to operate on pandas DataFrames and provide useful utilities for data preprocessing and cleaning.

"""


import math

import numpy as np
import pandas as pd
from scipy import stats


class DataFrameTransform:
    """
    A class containing various static methods for transforming data in pandas DataFrames. 
    
    Methods include data transformations such as dropping columns, handling missing values, 
    performing common statistical transformations, and removing outliers.
    """
    @staticmethod
    def drop_columns(df, column):
        """
        Drops the specified column from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which the column will be dropped.
            column (str): The name of the column to drop.

        Returns:
            pd.DataFrame: The DataFrame with the specified column removed.
        """
        df.drop(column, axis=1, inplace=True) 
        return df
    
    @staticmethod
    def drop_null_rows(df, column):
        """
        Drops rows from the DataFrame where the specified column contains null values.

        Args:
            df (pd.DataFrame): The DataFrame from which rows with null values in the specified column will be dropped.
            column (str): The column name to check for null values.

        Returns:
            pd.DataFrame: The DataFrame with rows containing null values in the specified column removed.
        """
        df.dropna(axis=0, subset=[column], inplace=True) 
        return df
    
    @staticmethod
    def calculate_loan_term(principal, interest_rate_annual, monthly_payment):
        """
        Calculate the loan term in months for a loan from its 
        princpal amount, annual interest rate and monthly payment amount.

        Args:
            principal (float): The principal loan amount.
            interest_rate_annual (float): Annual interest rate as a percentage.
            monthly_payment (float): Monthly payment amount.

        Returns:
            float: The number of months required to pay off the loan. 
                Returns float('inf') if the loan cannot be paid off,
                or NaN if any input is invalid.
        """
        # Check for missing or invalid values
        if any(val is None or pd.isna(val) for val in [principal, interest_rate_annual, monthly_payment]):
            return float('nan')    
        # Convert annual interest rate to monthly interest rate
        monthly_rate = interest_rate_annual / 12 / 100
        
        # Handle cases where monthly_payment is less than or equal to monthly interest
        if monthly_payment <= principal * monthly_rate:
            return float('nan')  # Indicate that the loan cannot be paid off
        
        # Calculate the number of months
        num_months = math.log(monthly_payment / (monthly_payment - principal * monthly_rate)) / math.log(1 + monthly_rate)
        return round(num_months)  # Round up to the nearest whole number

    @staticmethod
    def log_transform(df, column):
        """
        Applies a logarithmic transformation to the specified column of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be transformed.
            column (str): The name of the column to apply the logarithmic transformation to.

        Returns:
            pd.DataFrame: The DataFrame with the specified column transformed using the logarithm.
        """
        df[column] = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return df
    
    @staticmethod
    def box_cox_transform(df, column):
        """
        Applies a Box-Cox transformation to the specified column of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be transformed.
            column (str): The name of the column to apply the Box-Cox transformation to.

        Returns:
            pd.DataFrame: The DataFrame with the specified column transformed using the Box-Cox method.
        """
        df[column] = stats.boxcox(df[column])[0]
        return df
    
    @staticmethod
    def yeo_johnson_transform(df, column):
        """
        Applies a Yeo-Johnson transformation to the specified column of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be transformed.
            column (str): The name of the column to apply the Yeo-Johnson transformation to.

        Returns:
            pd.DataFrame: The DataFrame with the specified column transformed using the Yeo-Johnson method.
        """
        df[column] = stats.yeojohnson(df[column])[0]
        return df
    
    @staticmethod
    def remove_lower_outliers(df, column, lower_limit):
        """
        Removes rows from the DataFrame where the specified column has values below the given lower limit.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            column (str): The column name to check for values below the lower limit.
            lower_limit (float): The value below which data will be removed from the column.

        Returns:
            pd.DataFrame: The DataFrame with rows where the specified column value is below the lower limit removed.
        """
        new_df = df[df[column] >= lower_limit]
        return new_df
    
    @staticmethod
    def remove_upper_outliers(df, column, upper_limit):
        """
        Removes rows from the DataFrame where the specified column has values above the given upper limit.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            column (str): The column name to check for values above the upper limit.
            upper_limit (float): The value above which data will be removed from the column.

        Returns:
            pd.DataFrame: The DataFrame with rows where the specified column value is above the upper limit removed.
        """
        new_df = df[df[column] <= upper_limit]
        return new_df