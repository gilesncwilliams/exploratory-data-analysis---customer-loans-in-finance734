import pandas as pd
from sqlalchemy import create_engine, inspect
import yaml


def load_credentials():
    """
Reads credentials for an AWS RDS database from a yaml file.

Returns:
    credentials: a dictionary of the database credentials  
"""
    with open('credentials.yaml', 'r') as f:
        credentials = yaml.safe_load(f)
    return credentials


class RDSDatabaseConnector:

    def __init__(self, credentials):
        self.credentials = credentials

    def init_engine(credentials):
        """
        Initialises an sqlalchemy database engine.

        Args:
            credentials: a dictionary of the database credentials  
        
        Returns:
            engine: an sqlalchmey engine
        """            
        engine = create_engine(f"postgresql+psycopg2://{credentials['RDS_USER']}:{credentials['RDS_PASSWORD']}@{credentials['RDS_HOST']}:{credentials['RDS_PORT']}/{credentials['RDS_DATABASE']}")
        return engine    

def read_rds_table(table_name, engine):
    """Reads an AWS RDS database table.

    Args:
        table_name: the name of the table from the RDS database to extract.
        engine: the sqlalchemy connection engine for the RDS database.

    Returns:
        df: a Pandas dataframe.
    """
    df = pd.read_sql_table(table_name, engine)
    return df

def save_csv(df, filename):
    """Downloads and saves a Pandas Dataframe a CSV file.

    Args:
        df: the Pandas dataframe.
        filename: the filename given to the resulting csv file.

    Returns:
        df: a Pandas dataframe.
    """
    df.to_csv(filename, index=False, encoding='utf-8')

if __name__=="__main__": 
    credentials = load_credentials()
    engine = RDSDatabaseConnector.init_engine(credentials)
    loan_payments_df = read_rds_table('loan_payments', engine)
    save_csv(loan_payments_df, 'loan_payments.csv')

