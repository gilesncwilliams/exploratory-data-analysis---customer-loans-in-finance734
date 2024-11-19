import pandas as pd


loan_payments_df = pd.read_csv('loan_payments.csv')
loan_pyaments_shape = loan_payments_df.shape
print(loan_pyaments_shape)
print(loan_payments_df.info())
print(loan_payments_df.head(20))