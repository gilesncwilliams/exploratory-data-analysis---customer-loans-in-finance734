"""Functions for visualising data in pandas Dataframes.

Utilising various plots and graphs to visualise data from columns in Pandas 
dataframes for EDA purposes.

Typical usage example:

    data_visual = Plotter()
    heatmap_graph = data_visual.df_heatmap()
"""

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


class Plotter:
    """
    Functions for visualising data in pandas Dataframes.
    """

    @staticmethod
    def missing_values_matrix(df):
        """
        Visualise null values in a dataframe.

        Args: 
            df (pandas.Dataframe): The DataFrame containing the data.

        Returns:
            matplotlib.figure.Figure: A Matplotlib figure object representing the generated matrix.
        """
        return msno.matrix(df)
    
    @staticmethod
    def probability_distribution(df, column):
        """
        Visualise the probability distribution of a column in a pandas Dataframe as a bar plot.

        Args: 
            df (pandas.Dataframe): The DataFrame containing the data.
            column (pandas.Series): A column in the pandas Dataframe, df.

        Returns:
            matplotlib.figure.Figure: A Matplotlib figure object representing the plot.

        """
        sns.set(font_scale=0.5)
        probs = df[column].value_counts(normalize=True)
        sns.barplot(y=probs.values, x=probs.index)
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title('Discrete Probability Distribution')
        return plt.show()
    
    @staticmethod
    def df_heatmap(df):
        """
        Visualise a dataframe in a heatmap to check correlation between columns.

        Args: 
            df (pandas.Dataframe): The DataFrame containing the data.

        Returns:
            matplotlib.axes._subplots.AxesSubplot: 
                an AxesSubplot object representing the generated heatmap.

        """
        corr = df.select_dtypes('number').corr()
        return sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".1f")
    
    @staticmethod
    def heatmap(df, column1, column2):
        """
        Visualise in a heatmap the correlation between two specific columns 
        in a pandas dataframe.

        Args: 
            df (pandas.Dataframe): The DataFrame containing the data.
            column1 (pandas.Series): The first column in the pandas Dataframe, df.
            column2 (pandas.Series): The second column in the pandas Dataframe, df.

        Returns:
            matplotlib.axes._subplots.AxesSubplot: an AxesSubplot object representing the generated heatmap.

        """
        return sns.heatmap(df[[column1, column2]].corr(), annot=True, cmap='coolwarm')
    
    @staticmethod
    def density_plot(df, column):
        """
        Visualise the distribution of pandas Dataframe column as a histogram.

        Args: 
            df (pandas.Dataframe): The DataFrame containing the data.
            column (pandas.Series): A column in the pandas Dataframe, df.

        Returns:
            matplotlib.axes._subplots.AxesSubplot: 
                An AxesSubplot object representing the generated histogram.

        """
        return sns.histplot(data=df, y=column, kde=True)
            
    @staticmethod
    def histogram_kde(df, variable_list):
        """
        Creates a grid of histograms with kernel density estimation (KDE) for specified variables.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            variable_list (list): A list of column names in the DataFrame to plot.

        Returns:
            seaborn.axisgrid.FacetGrid: A FacetGrid object representing the grid of histograms with KDEs.
        """    
        sns.set_theme(font_scale=0.7)
        f = pd.melt(df, value_vars=variable_list)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
        return g

    @staticmethod
    def qq_plot(df, column):
        """
        Generates a Q-Q plot for the specified column to assess the normality of the data.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            column (str): The name of the column to generate the Q-Q plot for.

        Returns:
            statsmodels.graphics.gofplots.qqplot: The Q-Q plot object.
        """
        qq_plot = qqplot(df[column], scale=1 ,line='q', fit=True)
        return qq_plot

    @staticmethod
    def log_transform_plot(df, column):
        """
        Creates a histogram plot for the log-transformed values of the specified column and
        displays the skewness of the transformed data.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            column (str): The name of the column to log-transform and plot.

        Returns:
            matplotlib.axes.Axes: The Axes object with the log-transformed histogram plot.
        """
        log_column = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        t=sns.histplot(log_column,label="Log Transform Skewness: %.2f"%(log_column.skew()) )
        t.legend()
        return t

    @staticmethod
    def boxcox_transform_plot(df, column):
        """
        Creates a histogram plot for the Box-Cox transformed values of the specified column 
        and displays the skewness of the transformed data.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            column (str): The name of the column to Box-Cox transform and plot.

        Returns:
            matplotlib.axes.Axes: The Axes object with the Box-Cox transformed histogram plot.
        """
        boxcox_var = df[column]
        boxcox_var = stats.boxcox(boxcox_var)
        boxcox_var = pd.Series(boxcox_var[0])
        t=sns.histplot(boxcox_var,label="Box-Cox Skewness: %.2f"%(boxcox_var.skew()) )
        t.legend()
        return t

    @staticmethod
    def yeo_johnson_plot(df, column):
        """
        Creates a histogram plot for the Yeo-Johnson transformed values of the specified column
        and displays the skewness of the transformed data.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            column (str): The name of the column to Yeo-Johnson transform and plot.

        Returns:
            matplotlib.axes.Axes: The Axes object with the Yeo-Johnson transformed histogram plot.
        """
        yeojohnson_var = df[column]
        yeojohnson_var = stats.yeojohnson(yeojohnson_var)
        yeojohnson_var = pd.Series(yeojohnson_var[0])
        t=sns.histplot(yeojohnson_var,label="Yeo Johnson Skewness: %.2f"%(yeojohnson_var.skew()) )
        t.legend()
        return t

    @staticmethod
    def scatter_plot(df, column, outliers, name):
        """
        Creates a scatter plot for the specified column, highlighting outliers.

        Args:
            df (pandas.DataFrame): A pandas DataFrame.
            column (str): The column name to plot.
            outliers (list[bool]): A list of boolean values indicating outliers (True for outliers).
            name (matplotlib.axes.Axes): A Matplotlib Axes object where the plot will be drawn.

        Returns:
            matplotlib.axes.Axes: The updated Axes object with the scatter plot.
        """
        name.scatter(range(len(df)), df[column], c=['blue' if not x else 'red' for x in outliers])
        name.set_title(f'{column} with outliers highlighted (scatter plot)')
        name.set_xlabel('index')
        name.set_ylabel('value')
        return name


    @staticmethod
    def box_plot(df, column, name):
        """
        Creates a box plot for the specified column in the DataFrame and displays it on the given axes.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            column (str): The name of the column to plot in the box plot.
            name (matplotlib.axes.Axes): The Matplotlib Axes object where the plot will be drawn.

        Returns:
            matplotlib.axes.Axes: The updated Axes object with the box plot.
        """
        sns.boxplot(x=df[column], ax=name)
        name.set_title(f'{column} with outliers (box plot)')
        name.set_xlabel('value')
        return name


    @staticmethod
    def outliers_visuals(df, column, outliers):
        """
        Creates side-by-side visualizations of a scatter plot and a box plot to highlight outliers
        for a specified column in the DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            column (str): The name of the column to visualize.
            outliers (list[bool]): A list of boolean values indicating outliers (True for outliers).

        Returns:
            matplotlib.figure.Figure: The Figure object containing the scatter plot and box plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))    
        Plotter.scatter_plot(df, column, outliers, ax1)
        Plotter.box_plot(df, column, ax2)
        plt.tight_layout()
        return fig