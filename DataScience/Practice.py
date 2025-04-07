#202135835 정지호
import pandas as pd
import featuretools as ft

# set to see all columns
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


clients = pd.read_csv('data/clients.csv', parse_dates=['joined'])
loans = pd.read_csv('data/loans.csv', parse_dates=['loan_start', 'loan_end'])
payments = pd.read_csv('data/payments.csv', parse_dates=['payment_date'])

# declare empty entityset
es = ft.EntitySet(id='clients')

# .entity_from_dataframe,  had been changed
es = es.add_dataframe(dataframe_name='clients', dataframe=clients, index='client_id', time_index='joined')
es = es.add_dataframe(dataframe_name='loans', dataframe=loans, logical_types={'repaid': "Categorical"}, index='loan_id')
es = es.add_dataframe(dataframe_name='payments', dataframe=payments, logical_types={'missed': "Categorical"}, make_index=True, index='payment_id')

stats = loans.groupby('client_id')['loan_amount'].agg(['sum'])
stats.columns = ['total_loan_amount']

# Merge with the clients dataframe
stats_add_total = clients.merge(stats, left_on='client_id', right_index=True, how='left')

print(stats.head(10))
print(stats_add_total.head(10))