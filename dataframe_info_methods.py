import numpy as np


class DataFrameInfo:

    @staticmethod
    def df_info(df):
        return print(df.info())
    
    @staticmethod
    def describe_df(df):
        return print(df.describe())
    
    @staticmethod
    def describe_col(df, column):
        return print(df[column].describe())
    
    @staticmethod
    def df_shape(df):
        return print(df.shape)
    
    @staticmethod
    def num_unique(df):
        return print(df.nunique())
    
    @staticmethod
    def count_nulls(df):
        return print(df.isnull().sum()/len(df))

    @staticmethod
    def df_mean(df):
        return print(df.mean(numeric_only=True))
   
    @staticmethod
    def column_mean(df, column):
        return df[column].mean()
    
    @staticmethod
    def df_median(df):
        return print(df.median(numeric_only=True))
   
    @staticmethod
    def column_median(df, column):
        return df[column].median()
    
    @staticmethod
    def column_mode(df, column):
        return df[column].mode()

    @staticmethod
    def df_std_dev(df):
        return print(df.std(numeric_only=True))
   
    @staticmethod
    def column_std_dev(df, column):
        return print(df[column].std())
    
    @staticmethod
    def count_distinct_values(df, column):
        return print(df[column].value_counts())
    
    @staticmethod
    def df_skew(df):
        return (print(df.skew(numeric_only=True)))
    
    @staticmethod
    def col_skew(df,column):
        return (print(df[column].skew()))
    
    @staticmethod
    def measure_iqr(df, column):
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
        mean_col = np.mean(df[column])
        std_col = np.std(df[column])
        z_scores = (df[column] - mean_col) / std_col
        return z_scores

    @staticmethod
    def lower_bound(df, column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr

    @staticmethod
    def upper_bound(df, column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        return q3 + 1.5 * iqr