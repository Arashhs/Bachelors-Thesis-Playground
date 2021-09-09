from sqlalchemy import create_engine

from query_bias_correstion import query_bias_correction

dataset_dir = 'dataset'
database_name = 'simpson_database'
db_user = 'root'
db_pass = ''
db_host = 'localhost'


def main():
    # creating SQLAlchemy Engine instance
    mysql_engine = create_engine(f'mysql://{db_user}:{db_pass}@{db_host}/{database_name}')
    
    # creating the query_bias_correction instance
    qbc = query_bias_correction(mysql_engine)
    
    # printing the options    
    print("1- Initialize SQL-database tables from CSV datasets")
    option = input('Select option: ')
    
    # executing selected functionality
    if option == '1':
        qbc.init_sql_db(dataset_dir)
    elif option == '2':
        query = input('Query: ')
        qbc.correct_query(query)


if __name__ == '__main__':
    main()