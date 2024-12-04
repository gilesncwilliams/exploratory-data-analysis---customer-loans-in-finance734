import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

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
    
    # @staticmethod
    # def calc_z_score(df, column):
    #     mean_col = np.mean(df[column])
    #     std_col = np.std(df[column])
    #     z_scores = (df[column] - mean_col) / std_col
    #     return z_scores

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
        new_df = df[df[column] > lower_limit]
        return new_df
    
    @staticmethod
    def remove_upper_outliers(df, column, upper_limit):
        new_df = df[df[column] < upper_limit]
        return new_df

    # @staticmethod
    # def cap_outliers(df, column, lower_percentile, upper_percentile):
    #     lower_limit = np.percentile(df[column], lower_percentile)
    #     upper_limit = np.percentile(df[column], upper_percentile)
    #     return np.clip(df[column], lower_limit, upper_limit)
    
class Plotter:

    @staticmethod
    def missing_values_matrix(df):
        return msno.matrix(df)
    
    @staticmethod
    def probability_distribution(df, column):
        sns.set(font_scale=0.5)
        probs = df[column].value_counts(normalize=True)
        sns.barplot(y=probs.values, x=probs.index)
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title('Discrete Probability Distribution')
        return plt.show()
    
    @staticmethod
    def df_heatmap(df):
        corr = df.select_dtypes('number').corr()
        return sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".1f")
    
    @staticmethod
    def heatmap(df, column1, column2):
        return sns.heatmap(df[[column1, column2]].corr(), annot=True, cmap='coolwarm')
    
    @staticmethod
    def density_plot(df, column):
        return sns.histplot(data=df, y=column, kde=True)
    
    @staticmethod
    def bar_plot(df, column):
        probs = df[column].value_counts(normalize=True)
        sns.barplot(y=probs.values, x=probs.index)
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title('Discrete Probability Distribution')
        return plt.show()
        
    @staticmethod
    def histogram_kde(df, variable_list):
        sns.set(font_scale=0.7)
        f = pd.melt(df, value_vars=variable_list)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)

    @staticmethod
    def qq_plot(df, column):
        qq_plot = qqplot(df[column], scale=1 ,line='q', fit=True)
        plt.show()

    @staticmethod
    def log_transform_plot(df, column):
        log_column = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        t=sns.histplot(log_column,label="Log Transform Skewness: %.2f"%(log_column.skew()) )
        t.legend()

    @staticmethod
    def boxcox_transform_plot(df, column):
        boxcox_var = df[column]
        boxcox_var = stats.boxcox(boxcox_var)
        boxcox_var = pd.Series(boxcox_var[0])
        t=sns.histplot(boxcox_var,label="Box-Cox Skewness: %.2f"%(boxcox_var.skew()) )
        t.legend()

    @staticmethod
    def yeo_johnson_plot(df, column):
        yeojohnson_var = df[column]
        yeojohnson_var = stats.yeojohnson(yeojohnson_var)
        yeojohnson_var = pd.Series(yeojohnson_var[0])
        t=sns.histplot(yeojohnson_var,label="Yeo Johnson Skewness: %.2f"%(yeojohnson_var.skew()) )
        t.legend()

    @staticmethod
    def box_plot(df, column):
        sns.boxplot(y=df[column], color='lightgreen', showfliers=True)
        # sns.swarmplot(y=df[column], color='black', size=5)
        plt.title(f'Box plot with outliers')
        plt.show()

    @staticmethod
    def scatter_plot(df, column, outliers):
        plt.scatter(range(len(df)), df[column], c=['blue' if not x else 'red' for x in outliers])
        plt.title('Variable with Outliers Highlighted (Scatter Plot)')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()

    @staticmethod
    def scatter_plot_2(df, column, outliers, name):
        name.scatter(range(len(df)), df[column], c=['blue' if not x else 'red' for x in outliers])
        name.set_title('dfset with Outliers Highlighted (Scatter Plot)')
        name.set_xlabel('Index')
        name.set_ylabel('Value')

    @staticmethod
    def box_plot_2(df, column, name):
        sns.boxplot(x=df[column], ax=name)
        name.set_title('Dataset with Outliers (Box Plot)')
        name.set_xlabel('Value')