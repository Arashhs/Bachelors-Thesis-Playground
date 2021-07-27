import os
import pandas as pd
from sqlalchemy import create_engine, types

dataset_dir = 'dataset'
database_name = 'simpson_database'
db_user = 'root'
db_pass = ''
db_host = 'localhost'

# create SQL tables from CSV dataset
def init_sql_db(dataset_directory):
    csv_files = [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_directory)]
    dataset_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in csv_files]
    for i, dataset_name in enumerate(dataset_names):
        mysql_engine = create_engine(f'mysql://{db_user}:{db_pass}@{db_host}')
        mysql_engine.execute(f'CREATE DATABASE IF NOT EXISTS {database_name}')
        mysql_engine = create_engine(f'mysql://{db_user}:{db_pass}@{db_host}/{database_name}')

        df = pd.read_csv(csv_files[i],sep=',',quotechar='\'',encoding='utf8')
        df.to_sql(dataset_name,con=mysql_engine,index=False,if_exists='replace')


def main():
    init_sql_db(dataset_dir)
    print('Hey')



if __name__ == '__main__':
    main()