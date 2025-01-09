# Exploratory Data Analysis - Customer Loans in Finance


![Static Badge](https://img.shields.io/badge/Exploratory%20Data%20Analysis%20Project-Customer%20Loans%20In%20Finance-blue)

![Static Badge](https://img.shields.io/badge/A%20Data%20Training%20Project%20By-Giles%20Williams-blue)



## Table of Contents

- Introduction
- Installation Instructions
- Usage Instructions
- Project File Structure
- Licensing Information


## Introduction

![Static Badge](https://img.shields.io/badge/Project%20Summary%3A-blue)

The project goal is to gain a comprehensive understanding of the loan portfolio data for a large financial instituion and, by performing exploratory data analysis on the loan portfolio data, to improve the performance and profitability of the loan portfolio.


![Static Badge](https://img.shields.io/badge/Project%20Scope%3A-blue)

By using various statistical and data visualisation techniques, the aim is to uncover patterns, relationships, and anomalies in the loan data, that enable the business to make more infomred decisions about loan approvals, pricing, and risk management in the future. 

The project is divided into two parts:

Part One: Exploratory Data Analysis, covers the main EDA tasks related to analysing the portfolio. Reviewing and cleaning the loan data to identify any issues, such as missing values, and then applying statitical techniques to gain insight into the data's distrbution in preperation for further statistical analysis or machine learning. 

Part Two: Analysis and Visualisation, drills down further into the data to draw insights and patterns that senior management can use to make more informed decisions about the current state of the loans, and highlighting which are at a higher risk to the company. 


## Installation Instructions

1. The first step to begin the installation process is to clone this repository to your local machine

    Follow [these](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository#cloning-a-repository) instructions on how to clone a repository.


2. Environment setup: using a package manager e.g Conda to manually import each package, or clone the full enviroment from this repository.

Ensure you are using a new environment before starting the set up process. 

Next, using your package manager, install Python3.

To follow the extraction, load and transform tasks in the Juypter Notebooks in the project, you will require these 3rd party packages installed into your environmnet. Using pip to install them:

- matplotlib
- numpy
- pandas
- scipy
- seaborn
- sqlalchemy
- statsmodels.graphics.gofplots
- yaml

Alternatively, you can use the eda_environment_export.yml file included in the project files to import the full project environment. If using the Condas package manager you can do so with the following command:

conda env create -f eda_environment_export.yml

See detailed instructions [here](https://github.com/conda/conda/blob/main/docs/source/user-guide/tasks/manage-environments.rst) on how to create a new environment and import an environment from a yaml file. 

3. Extracting the Loan data.

The loan data used in the project is stored in a AWS RDS database.
Run the db_utils.py script in the project foler that will connect to the RDS database, extract the data from the required table and download it to your local machine as a csv file.  


4. Project Orientation. 

The main detail of the project is contained within the two Juypter Notebooks in the project folder.

Each part of the project has its own notebook:

- Part One: Part One Exploratory Data Analysis = eda_loan_payments.ipynb
- Part Two: Analysis and Visualisation = analysis_visualisation_loan_payments.ipynb

Before opening either of the notebooks, be sure you are in the correct environment containing the above mentioned packages. Once opened, you should the run all the cells in the notebook to view all the findings and visualisations.

Important: Part Two of the project is based on a snapshot of the data midway through Part One, following the data cleaning, and prior to any skew, or outlier transforms that take place. So please ensure you run the notebook for Part One first, as this will download the requried csv file to your local machine that is used in Part Two's notebook. The file downloaded is called loan_payments_nulls_removed.csv. A copy of which is also saved in the repository.


## Project File Structure

 * exploratory-data-analysis---customer-loans-in-finance734
   * analysis_visualisation_loan_payments.ipynb
   * credentials.yaml
   * data_transform_methods.py
   * data_visualisation_methods.py
   * dataframe_info_methods.py
   * dataframe_transformation_methods.py
   * db_utils.py
   * eda_loan_payments.ipynb
   * loan_payments_nulls_removed.csv
   * loan_payments.csv
   * README.md
   * .gitignore  


## License Information
This repo is unlicensed as it was intended only for training purposes.