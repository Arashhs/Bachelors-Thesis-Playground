import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from query_bias_correstion import query_bias_correction

dataset_dir = 'dataset'
database_name = 'simpson_database'
db_user = 'root'
db_pass = ''
db_host = 'localhost'

queries_file = 'biased_queries.txt'


# comparing the performance of the two proposed methods
def compare_performance(qbc, queries_file):
    # reading all of the queries
    with open(queries_file, 'r') as file:
        lines = [line.strip() for line in file]
        queries = list(filter(lambda q: q!='', lines))
    # initializing run-times
    columns = ['target_tables', 'first_met_perfs', 'second_met_perfs', 'is_equal']
    data = []
    # appending run-times
    for query in queries:
        target_table = query.lower().split('from')[1].split()[0].replace('`', '')
        first, second, is_equal = qbc.compare_performance(query)
        data.append([target_table, first, second, is_equal])
    df = pd.DataFrame(data, columns=columns)
    fig, ax = plt.subplots(1, 1)
    df = df.set_index('target_tables')
    ax = df.plot(legend=True, ax=ax)
    for col in columns[1:3]:
        for id, val in enumerate(df[col]):
            val = float("{:.3f}".format(val))
            ax.text(id, val, str(val))
    plt.title('Comparing Performance of the Two Proposed Methpds', fontsize=16, fontweight='bold')
    plt.xlabel(f'Dataset Name', fontsize=14)
    plt.ylabel(f'Total spent time (Seconds)', fontsize=14)
    plt.show()
    



def main():
    # creating SQLAlchemy Engine instance
    mysql_engine = create_engine(f'mysql://{db_user}:{db_pass}@{db_host}/{database_name}')
    
    # creating the query_bias_correction instance
    qbc = query_bias_correction(mysql_engine)
    
    # printing the options    
    print("1- Initialize SQL-database tables from CSV datasets")
    print("2- Detect and correct query bias the first algorithm (direct method)")
    print("3- Detect and correct query bias the second algorithm (using regression)")
    print("4- Analyse the dataset and find the Simpson pairs (using direct method)")
    print("5- Analyse the dataset and find the Simpson pairs (using regression)")
    print('6- Compare the performance of the two algorithms (direct method vs regression-based method)')


    option = input('Select option: ')
    
    # executing selected functionality
    if option == '1':
        mysql_engine = create_engine(f'mysql://{db_user}:{db_pass}@{db_host}')
        mysql_engine.execute(f'CREATE DATABASE IF NOT EXISTS {database_name}')
        qbc.init_sql_db(dataset_dir)
    elif option == '2':
        query = input('Query: ')
        print(qbc.process_query(query))
    elif option == '3':
        query = input('Query: ')
        print(qbc.process_query_regression(query))
    elif option == '4':
        query = input('Query: ')
        qbc.analyse_dataset(query)
    elif option == '5':
        query = input('Query: ')
        qbc.analyse_dataset_regression(query)
    elif option == '6':
        compare_performance(qbc, queries_file)


if __name__ == '__main__':
    main()