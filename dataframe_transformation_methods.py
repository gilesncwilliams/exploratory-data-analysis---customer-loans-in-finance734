import math
import numpy as np
import pandas as pd
from scipy import stats


class DataFrameTransform:

    @staticmethod
    def drop_columns(df, column):
        df.drop(column, axis=1, inplace=True) 
        return df
    
    @staticmethod
    def drop_null_rows(df, column):
        df.dropna(axis=0, subset=[column], inplace=True) 
        return df
    
    @staticmethod
    def calculate_loan_term(principal, interest_rate_annual, monthly_payment):
        """
            Calculate the loan term in months for given loan details.

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
        df[column] = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        return df
    
    @staticmethod
    def box_cox_transform(df, column):
        df[column] = stats.boxcox(df[column])[0]
        return df
    
    @staticmethod
    def yeo_johnson_transform(df, column):
        df[column] = stats.yeojohnson(df[column])[0]
        return df
    
    @staticmethod
    def remove_lower_outliers(df, column, lower_limit):
        new_df = df[df[column] >= lower_limit]
        return new_df
    
    @staticmethod
    def remove_upper_outliers(df, column, upper_limit):
        new_df = df[df[column] <= upper_limit]
        return new_df