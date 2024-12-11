import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


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
        sns.set_theme(font_scale=0.7)
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
    def scatter_plot(df, column, outliers, name):
        name.scatter(range(len(df)), df[column], c=['blue' if not x else 'red' for x in outliers])
        name.set_title(f'{column} with outliers highlighted (scatter plot)')
        name.set_xlabel('index')
        name.set_ylabel('value')

    @staticmethod
    def box_plot(df, column, name):
        sns.boxplot(x=df[column], ax=name)
        name.set_title(f'{column} with outliers (box plot)')
        name.set_xlabel('value')

    @staticmethod
    def outliers_visuals(df, column, outliers):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))    
        Plotter.scatter_plot(df, column, outliers, ax1)
        Plotter.box_plot(df, column, ax2)
        plt.tight_layout()
        plt.show()