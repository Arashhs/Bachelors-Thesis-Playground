import os
import math
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, types
import sqlparse

dataset_dir = 'dataset'
database_name = 'simpson_database'
db_user = 'root'
db_pass = ''
db_host = 'localhost'

bias_degree_threshold = 0.3

# create SQL tables from CSV dataset
def init_sql_db(dataset_directory):
    csv_files = [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_directory)]
    dataset_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in csv_files]
    for i, dataset_name in enumerate(dataset_names):
        mysql_engine = create_engine(f'mysql://{db_user}:{db_pass}@{db_host}')
        mysql_engine.execute(f'CREATE DATABASE IF NOT EXISTS {database_name}')
        mysql_engine = create_engine(f'mysql://{db_user}:{db_pass}@{db_host}/{database_name}')

        # df = pd.read_csv(csv_files[i],sep=',',quotechar='\'',encoding='utf8')
        df = pd.read_csv(csv_files[i])
        df.to_sql(dataset_name,con=mysql_engine,index=False,if_exists='replace')
    return df


# getting normalized of a vector
def get_normalized(vector):
    vec_len = sum([v**2 for v in vector])
    vec_len = math.sqrt(vec_len)
    if vec_len == 0:
        return vector
    norm_vec = [v/vec_len for v in vector]
    return norm_vec


# calculating cosine score between two vectors
def get_cosine_score(vec1, vec2):
    v1 = get_normalized(vec1)
    v2 = get_normalized(vec2)
    cosine_sim = np.dot(v1, v2)
    return cosine_sim


# get the bias degree
def get_bias_degree(vec1, vec2):
    return 1 - get_cosine_score(vec1, vec2)


# get the desired groups based on common attributes in dataset
def get_groups(dataframe, common_att_list):
    splited_df = dataframe.groupby(common_att_list)
    common_att_vals = list(splited_df.groups.keys())
    groups = []
    for att_vals in common_att_vals:
        groups.append(splited_df.get_group(att_vals))
    return groups, common_att_vals


# get the mean of the desired attribute for several groups
def get_groups_avgs(groups, target_attribute):
    avgs = [group[target_attribute].mean() for group in groups]
    return avgs


'''
# get the probability than Simpson's paradox has happened
def get_paradox_prob(agg_avg, de_agg_avgs):
    first = np.sum([avg > agg_avg for avg in de_agg_avgs])
    second = np.sum([avg < agg_avg for avg in de_agg_avgs])
    num_of_reverses = max(first, second)
    return num_of_reverses/len(de_agg_avgs)
'''


# check if the Simpson's paradox has happened based on two set of groups
def is_biased(first_gp_avgs, second_gp_avgs):
    agg_attribute_vals = list(first_gp_avgs.keys())
    test_attribute_vals = np.unique([val[1] for val in second_gp_avgs.keys()])
    agg_bigger_num = []
    test_bigger_num = []
    for agg_att_val in agg_attribute_vals:
	    agg_bigger_num.append((np.sum([first_gp_avgs[agg_att_val] >= first_gp_avgs[val] for val in agg_attribute_vals]) - 1))
    for agg_att_val in agg_attribute_vals:
        current_count = 0
        for test_att_val in test_attribute_vals:
            if (agg_att_val, test_att_val) not in second_gp_avgs:
                continue
            current_count += np.sum([second_gp_avgs[agg_att_val, test_att_val] >= second_gp_avgs[val, test_att_val] for val in agg_attribute_vals if (val, test_att_val) in second_gp_avgs]) - 1
        test_bigger_num.append(current_count)
    bias_degree = get_bias_degree(agg_bigger_num, test_bigger_num)
    # if test_bigget_num is all zeroes or the bias degree is lower than the threshold, return False
    if np.all(np.array(test_bigger_num) == 0) or bias_degree < bias_degree_threshold:
        return False
    return True


# Check if the averages of target attribute reverse if we de-aggregate data using test_attribute
def check_for_bias(dataframe, agg_attribute, test_attribute, target_attribute):
    first_groups, first_att_vals = get_groups(dataframe, [agg_attribute])
    second_groups, second_att_vals  = get_groups(dataframe, [agg_attribute, test_attribute])
    first_avgs = get_groups_avgs(first_groups, target_attribute)
    second_avgs = get_groups_avgs(second_groups, target_attribute)
    first_group_avgs = {key:value for key, value in zip(first_att_vals, first_avgs)}
    second_group_avgs = {key:value for key, value in zip(second_att_vals, second_avgs)}
    return is_biased(first_group_avgs, second_group_avgs)

'''
# Check if query is prone to the Simpson's paradox
def is_prone_to_bias(query):
    regex = r'^(SELECT)\s.*(AVG\s*\(.*\)).*\s(GROUP\s+BY)\s+[\w]+$'
    if re.search(regex, query, re.IGNORECASE):
        return True
    return False '''


def get_target_from_token(token, target_attributes):
    avg_seen = False
    for sub_token in token.tokens:
        if avg_seen:
            for par_token in sub_token.tokens:
                if isinstance(par_token, sqlparse.sql.IdentifierList):
                    for identifier in par_token.get_identifiers():
                        print("{} {}\n".format("TargetAtt = ", identifier) )
                        target_attributes.append(identifier.value.lower())
                elif isinstance(par_token, sqlparse.sql.Identifier):
                    target_attributes.append(par_token.value.lower())
                    print("{} {}\n".format("TargetAtt = ", par_token) )
            avg_seen = False
        if sub_token.value.upper() == "AVG":
            avg_seen = True
    return target_attributes


def extract_needed_features(query):
    select_seen, avg_seen, from_seen, group_by_seen = (False, False, False, False)
    target_attributes = []
    agg_attributes = []
    tables = []
    columns = []
    parsed_query = sqlparse.parse(query)[0]
    for token in parsed_query:
        if select_seen:
            if isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    if isinstance(identifier, sqlparse.sql.Function):
                        get_target_from_token(identifier, target_attributes)
                        continue
                    columns.append(identifier.value.lower())
                    print("{} {}\n".format("Attr = ", identifier) )
            elif isinstance(token, sqlparse.sql.Identifier):
                columns.append(token.value.lower())
                print("{} {}\n".format("Attr = ", token))
            elif isinstance(token, sqlparse.sql.Function):
                get_target_from_token(token, target_attributes)
        if from_seen:
            if isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    tables.append(identifier.value.lower())
                    print("{} {}\n".format("TAB = ", identifier) )
            elif isinstance(token, sqlparse.sql.Identifier):
                tables.append(token.value.lower())
                print("{} {}\n".format("TAB = ", token))
        if group_by_seen:
            if isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    agg_attributes.append(identifier.value.lower())
                    print("{} {}\n".format("GROUPBY att = ", identifier) )
            elif isinstance(token, sqlparse.sql.Identifier):
                agg_attributes.append(token.value.lower())
                print("{} {}\n".format("GROUPBY att = ", token))
            
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "GROUP BY":
            select_seen = False
            from_seen = False
            group_by_seen = True
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
            select_seen = False
            from_seen = True
            group_by_seen = False
        if token.ttype is sqlparse.tokens.DML and token.value.upper() == "SELECT":
            select_seen = True
            from_seen = False
            group_by_seen = False
        
    return target_attributes, tables, agg_attributes, columns


# Check whether Simpson's paradox is present and correct it if present
def detect_and_correct_execute(query, sql_engine, target_attributes, tables, agg_attributes, columns):
    df = pd.read_sql_table(tables[0], sql_engine)
    df.columns = df.columns.str.lower()
    for test_attribute in df.columns:
        if test_attribute not in agg_attributes and test_attribute not in target_attributes:
            print(test_attribute, check_for_bias(df, agg_attributes[0], test_attribute, target_attributes[0]))


# process query, detect and correct Simpson's paradox if present
def process_query(query, sql_engine):
    target_attributes, tables, agg_attributes, columns = extract_needed_features(query)
    if len(target_attributes) > 0 and len(agg_attributes) > 0:
        detect_and_correct_execute(query, sql_engine, target_attributes, tables, agg_attributes, columns)
    else:
        sql_engine.execute(query)


def main():
    mysql_engine = create_engine(f'mysql://{db_user}:{db_pass}@{db_host}/{database_name}')
    print("1- Initialize SQL-database tables from CSV datasets")
    option = input('Select option: ')
    if option == '1':
        init_sql_db(dataset_dir)
    elif option == '2':
        query = input('Query: ')
        # query = 'SELECT avg(admit) from admissions_data GROUP BY Gender'
        print(process_query(query, mysql_engine))
    # check_for_bias(df, 'Treated', 'Treated', 'Survived')
    # check_for_avg_reverse(df, 'Gender', 'Dept', 'Admit')
    # check_for_avg_reverse(df, 'player', 'year', 'outcome')



if __name__ == '__main__':
    main()