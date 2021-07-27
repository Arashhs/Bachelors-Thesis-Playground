import os
import numpy as np
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
    return df


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


# get the probability than Simpson's paradox has happened
def get_paradox_prob(agg_avg, de_agg_avgs):
    first = np.sum([avg > agg_avg for avg in de_agg_avgs])
    second = np.sum([avg < agg_avg for avg in de_agg_avgs])
    num_of_reverses = max(first, second)
    return num_of_reverses/len(de_agg_avgs)


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
    if agg_bigger_num.index(max(agg_bigger_num)) > test_bigger_num.index(max(test_bigger_num)) and max(test_bigger_num) != 0:
        return True
    return False
    

    


# Check if the averages of target attribute reverse if we de-aggregate data using test_attribute
def check_for_avg_reverse(dataframe, agg_attribute, test_attribute, target_attribute):
    first_groups, first_att_vals = get_groups(dataframe, [agg_attribute])
    second_groups, second_att_vals  = get_groups(dataframe, [agg_attribute, test_attribute])
    first_avgs = get_groups_avgs(first_groups, target_attribute)
    second_avgs = get_groups_avgs(second_groups, target_attribute)
    first_group_avgs = {key:value for key, value in zip(first_att_vals, first_avgs)}
    second_group_avgs = {key:value for key, value in zip(second_att_vals, second_avgs)}
    print(is_biased(first_group_avgs, second_group_avgs))



def main():
    df = init_sql_db(dataset_dir)
    check_for_avg_reverse(df, 'Treated', 'Gender', 'Survived')
    # check_for_avg_reverse(df, 'Gender', 'Dept', 'Admit')
    # check_for_avg_reverse(df, 'player', 'year', 'outcome')



if __name__ == '__main__':
    main()