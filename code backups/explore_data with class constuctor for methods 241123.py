import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

class DataTransform:
    
    def __init__(self, df):
        self.df = df

    def transform_to_category(self, column):
        self.df[column] = self.df[column].astype('category')
        return self.df[column]
    
    def transform_to_boolean(self, column):
        self.df[column] = self.df[column].map({'y': True, 'n': False})
        return self.df[column]
    
    def transform_to_datetime(self, column):
        self.df[column] = pd.to_datetime(self.df[column], errors='coerce', format='mixed')
        return self.df[column]

# initial check of data
# loan_payments_df = pd.read_csv('loan_payments.csv')
# loan_payments_shape = loan_payments_df.shape
# print(loan_payments_shape)
# print(loan_payments_df.info())
# pd.set_option("display.max_columns", None)
# print(loan_payments_df.head(20))

class DataFrameInfo:

    def __init__(self, df):
        self.df = df
    
    def df_info(self):
        return print(self.df.info())
    
    def describe_df(self):
        return print(self.df.describe())
    
    def df_shape(self):
        return print(self.df.shape)
    
    def num_unique(self):
        return print(self.df.nunique())
    
    def count_nulls(self):
        return print(self.df.isnull().sum()/len(self.df))

    def df_mean(self):
        return print(self.df.mean(numeric_only=True))
   
    def column_mean(self, column):
        return self.df[column].mean()
    
    def df_median(self):
        return print(self.df.median(numeric_only=True))
   
    def column_median(self, column):
        return self.df[column].median()
    
    def column_mode(self, column):
        return self.df[column].mode()

    def df_std_dev(self):
        return print(self.df.std(numeric_only=True))
   
    def column_std_dev(self, column):
        return print(self.df[column].std())
    
    def count_distinct_values(self, column):
        return print(self.df[column].value_counts())
    
class DataFrameTransform:

    def __init__(self, df):
        self.df = df

    def drop_columns(self, column):
        return self.df.drop(column, axis=1) 

class Plotter:

    def __init__(self, df):
        self.df = df

    def missing_values_matrix(self):
        return msno.matrix(self.df)
    
    def probability_distribution(self, column):
        sns.set(font_scale=0.5)
        probs = self.df[column].value_counts(normalize=True)
        sns.barplot(y=probs.values, x=probs.index)
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title('Discrete Probability Distribution')
        return plt.show()
    
    def heatmap(self, column1, column2):
        return sns.heatmap(self.df[[column1, column2]].corr(), annot=True, cmap='coolwarm')
    
    def density_plot(self, column):
        return sns.histplot(data=self.df, y=column, kde=True)
    
    def bar_plot(self, column):
        probs = self.df[column].value_counts(normalize=True)
        sns.barplot(y=probs.values, x=probs.index)
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title('Discrete Probability Distribution')
        return plt.show()
