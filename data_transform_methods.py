import pandas as pd


class DataTransform:
    
    @staticmethod
    def transform_to_category(df, column):
        df[column] = df[column].astype('category')
        return df[column]
    
    @staticmethod
    def transform_to_boolean(df, column):
        df[column] = df[column].map({'y': True, 'n': False})
        return df[column]
    
    @staticmethod
    def transform_to_datetime(df, column):
        df[column] = pd.to_datetime(df[column], errors='coerce', format='mixed')
        return df[column]

    @staticmethod
    def transform_to_numeric(df, column):
        df[column] = pd.to_numeric(df[column], errors='coerce')
        return df[column]