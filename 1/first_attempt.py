import os
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.arrays import categorical
from sqlalchemy import create_engine, types
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
import cvxopt

import statsmodels.api as sm
import seaborn as sns
import sqlparse



dataset_dir = 'dataset'
database_name = 'simpson_database'
db_user = 'root'
db_pass = ''
db_host = 'localhost'

bias_degree_threshold = 0.3
groups_num_threshold = 100

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


# encode dataset with categorical data
def encode_categorical_cols(df):
    categorical_cols = [col for col in df.columns if df[col].dtype=="O"]
    for cat_col in categorical_cols:
        df[cat_col] = df[cat_col].astype('category').cat.codes
    return df, categorical_cols


# getting normalized of a vector
def get_normalized(vector):
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0: 
       return vector
    return vector / norm


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


# add key-pairs with the the value 0 if key-pair does not exist in dict
def augment_zeroes(group_avgs, agg_vals, test_vals):
    for test_val in test_vals:
        for agg_val in agg_vals:
            if (agg_val, test_val) not in group_avgs:
                group_avgs[agg_val, test_val] = 0
    return group_avgs


# check if the Simpson's paradox has happened based on two set of groups
def is_biased(first_gp_avgs, second_gp_avgs):
    agg_attribute_vals = list(first_gp_avgs.keys())
    test_attribute_vals = np.unique([val[1] for val in second_gp_avgs.keys()])
    second_gp_avgs = augment_zeroes(second_gp_avgs, agg_attribute_vals, test_attribute_vals)
    agg_bigger_num = []
    test_bigger_num = []
    for agg_att_val in agg_attribute_vals:
	    agg_bigger_num.append((np.sum([first_gp_avgs[agg_att_val] >= first_gp_avgs[val] for val in agg_attribute_vals]) - 1))
    test_bigger_num = 0 * np.array(agg_bigger_num)
    for test_att_val in test_attribute_vals:
        current_vec = []
        for agg_att_val in agg_attribute_vals:
            current_vec.append((np.sum([second_gp_avgs[agg_att_val, test_att_val] >= second_gp_avgs[val, test_att_val] for val in agg_attribute_vals]) - 1))
        test_bigger_num = np.add(test_bigger_num, current_vec)
    test_bigger_num = list(test_bigger_num)
    # bias_degree = get_bias_degree(agg_bigger_num, test_bigger_num)
    # if np.all(np.array(test_bigger_num) == 0):
    #     bias_degree = 0
    # if test_bigget_num is all zeroes or the bias degree is lower than the threshold, return False
    # if bias_degree < bias_degree_threshold:
    #     return False, bias_degree
    # return True, bias_degree
    is_biased, bias_degree = check_if_biased(agg_bigger_num, test_bigger_num)
    return is_biased, bias_degree



# Check if the averages of target attribute reverse if we disaggregate data using test_attribute
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


# Only hold essential parts of query for retreiving dataframe
def preprocess_query(query):
    unwanted_keywords = ['SELECT', 'GROUP', 'ORDER', 'HAVING']
    seen_unwanted = False
    formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
    sql_lines = formatted_query.split('\n')
    manipulated_sql = 'SELECT *'
    for line in sql_lines:
        if line.startswith((' ', '\t')):
            print(f'param-line: {line}')
            print(f"parameter: {line.strip().replace(r',', '')}")
            print()
        else:
            print(f'Statement: {line}')
            splitted_stmt = line.split()
            keyword = splitted_stmt[0]
            if keyword.upper() in unwanted_keywords:
                seen_unwanted = True
            else:
                seen_unwanted  = False
            print(f'Keyword: {splitted_stmt[0]}')
            print(f"Param: {splitted_stmt[1].strip().replace(r',', '')}")
            print()
        
        if not seen_unwanted:
            manipulated_sql += '\n' + line
    print(f'manipulated_sql: {manipulated_sql}')
    return manipulated_sql


# correct the biased query by disaggregating by a candidate attribute
def correct_query(query, disaggregated_att):
    target_keywords = ['SELECT', 'GROUP']
    seen_target = False
    formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
    sql_lines = formatted_query.split('\n')
    corrected_query = ''
    for line in sql_lines:
        if line.startswith((' ', '\t')):
            print(f'param-line: {line}')
            print(f"parameter: {line.strip().replace(r',', '')}")
            print()
        else:
            if seen_target:
                corrected_query += f', `{disaggregated_att}`'
            print(f'Statement: {line}')
            splitted_stmt = line.split()
            keyword = splitted_stmt[0]
            if keyword.upper() in target_keywords:
                seen_target = True
            else:
                seen_target  = False
            print(f'Keyword: {splitted_stmt[0]}')
            print(f"Param: {splitted_stmt[1].strip().replace(r',', '')}")
            print()
        corrected_query += '\n' + line
    if seen_target:
        corrected_query += f', `{disaggregated_att}`'
    corrected_query = sqlparse.format(corrected_query, reindent=True, keyword_case='upper')
    print(f'manipulated_sql: {corrected_query}')
    return corrected_query


# Check whether Simpson's paradox is present and correct it if present
def detect_and_correct_query_bias(query, sql_engine, target_attributes, tables, agg_attributes, columns):
    df = pd.read_sql(preprocess_query(query), sql_engine)
    df.columns = df.columns.str.lower()
    biased_degrees_dict = dict()
    unbiased_degrees_dict = dict()
    for test_attribute in df.columns:
        if test_attribute not in agg_attributes and test_attribute not in target_attributes and len(df[test_attribute].unique()) < groups_num_threshold:
            is_query_biased, bias_degree = check_for_bias(df, agg_attributes[0], test_attribute, target_attributes[0])
            print(test_attribute, is_query_biased, bias_degree)
            if is_query_biased:
                biased_degrees_dict[test_attribute] = bias_degree
            else:
                unbiased_degrees_dict[test_attribute] = bias_degree
    if len(biased_degrees_dict) > 0:
        # Query was biased
        # Select the attribute for which the bias degree was highest
        best_test_att, highest_bias_degree = max(biased_degrees_dict.items(), key=lambda v: v[1])
        corrected_query = correct_query(query, best_test_att)
        plot_query_results(query, corrected_query, sql_engine, agg_attributes[0], best_test_att, target_attributes[0])
        return corrected_query
    else:
        # Query was unbiased
        return query


# process query, detect and correct Simpson's paradox if present
def process_query(query, sql_engine):
    target_attributes, tables, agg_attributes, columns = extract_needed_features(query)
    target_attributes = [att.replace('`', '') for att in target_attributes]
    tables = [table.replace('`', '') for table in tables]
    agg_attributes = [att.replace('`', '') for att in agg_attributes]
    columns = [att.replace('`', '') for att in columns]
    if len(target_attributes) > 0 and len(agg_attributes) > 0:
        return detect_and_correct_query_bias(query, sql_engine, target_attributes, tables, agg_attributes, columns)
    else:
        return query


# analyse dataset to see if Simpson's paradox is present
def analyse_dataset(query, sql_engine):
    df = pd.read_sql(preprocess_query(query), sql_engine)
    df, categorical_cols = encode_categorical_cols(df)
    for target_att in (set(df.columns) - set(categorical_cols)):
        for agg_att in df.columns:
            if agg_att != target_att:
                for test_att in df.columns:
                    if (test_att != target_att) and (test_att != agg_att):
                        biased, score = check_for_bias(df, agg_att, test_att, target_att)
                        if biased:
                            print(check_for_bias(df, agg_att, test_att, target_att), f"target:{target_att}, agg:{agg_att}, test:{test_att}")



def plot_query_results(query, corrected_query, sql_engine, agg_att, disagg_att, target_att):
    df_biased = pd.read_sql(query, sql_engine)
    df_biased.columns = df_biased.columns.str.lower().str.replace('`', '')
    df_corrected = pd.read_sql(corrected_query, sql_engine)
    df_corrected.columns = df_corrected.columns.str.lower().str.replace('`', '')
    # plotting biased query results
    ax = df_biased.groupby([agg_att])['avg(%s)' % target_att].apply(float).plot(kind='bar', rot=1, legend=False, color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
    plt.title('Biased Query Results', fontsize=16, fontweight='bold')
    plt.xlabel(f'{agg_att.capitalize()}', fontsize=14)
    plt.ylabel(f'AVG({target_att.capitalize()})', fontsize=14)
    # annotating the chart
    for p in ax.patches:
        # ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # plotting corrected query results
    ax = df_corrected.groupby([disagg_att, agg_att])['avg(%s)' % target_att].apply(float).unstack().plot(kind='bar', rot=1)
    plt.title('Corrected Query Results', fontsize=16, fontweight='bold')
    plt.xlabel(f'{disagg_att.capitalize()}', fontsize=14)
    plt.ylabel(f'AVG({target_att.capitalize()})', fontsize=14)
    # annotating the chart
    for p in ax.patches:
        # ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')


############ THE SECOND METHOD - USING REGRESSION ############
# process query, detect and correct Simpson's paradox using regression models if present
def process_query_regression(query, sql_engine, ignore_features):
    target_attributes, tables, agg_attributes, columns = extract_needed_features(query)
    target_attributes = [att.replace('`', '') for att in target_attributes]
    tables = [table.replace('`', '') for table in tables]
    agg_attributes = [att.replace('`', '') for att in agg_attributes]
    columns = [att.replace('`', '') for att in columns]
    if len(target_attributes) > 0 and len(agg_attributes) > 0:
        return detect_and_correct_query_bias_regression(query, sql_engine, target_attributes, tables, agg_attributes, columns, ignore_features)
    else:
        return query


# Check whether Simpson's paradox is present and correct it using regression models if present
def detect_and_correct_query_bias_regression(query, sql_engine, target_attributes, tables, agg_attributes, columns, ignore_features):
    df = pd.read_sql(preprocess_query(query), sql_engine)
    biased_degrees_dict, unbiased_degrees_dict = build_regression_models(df, target_attributes[0], agg_attributes[0], ignore_features)
    if len(biased_degrees_dict) > 0:
        # Query was biased
        # Select the attribute for which the bias degree was highest
        best_test_att, highest_bias_degree = max(biased_degrees_dict.items(), key=lambda v: v[1])
        corrected_query = correct_query(query, best_test_att)
        plot_query_results(query, corrected_query, sql_engine, agg_attributes[0], best_test_att, target_attributes[0])
        plot_reg_results(df, target_attributes[0], agg_attributes[0], best_test_att)
        return corrected_query
    else:
        # Query was unbiased
        return query


# build a regression model and return parameters
def build_regression_model(data, response_feature, predictor_feature):
    Y = data[response_feature]
    X = data[predictor_feature]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    params = model.params

    # plot
    # x_values = np.arange(data[predictor_feature].min(), data[predictor_feature].max() + 0.5, 0.5)
    # scatter-plot data
    # ax = data.plot(x=predictor_feature, y=response_feature, kind='scatter')
    # plot regression line on the same axes, set x-axis limits
    # ax.plot(x_values, params.const + params[predictor_feature] * x_values)
    return params


# build a logistic regression model and return parameters
def build_logistic_regression_model(data, response_feature, predictor_feature, categorical_cols):
    Y = data[response_feature]
    X = data[predictor_feature]
    if predictor_feature in categorical_cols:
        X = pd.get_dummies(X)
    if len(X) < 2:
        return
    result = LogisticRegression(random_state=0).fit(X, Y)
    return result.coef_


# build representation vector from regression model
def build_reg_represent_vector(slopes, predictor_feature):
    vec = [0, 0]
    for slope in slopes:
        predictor_slope = slope[predictor_feature]
        if predictor_slope >= 0:
            vec[0] += 1
        else:
            vec[1] += 1
    return vec


# check if the regression representations v1 and v2 suggest biased query
def check_if_biased(v1, v2):
    ind_1 = v1.index(max(v1))
    ind_2 = v2.index(max(v2))
    bias_degree = get_bias_degree(v1, v2)
    if ind_1 != ind_2 and not np.all(np.array(v2) == 0):
        return True, bias_degree
    return False, bias_degree




# build a regression model using given inputs
def build_regression_models(df, response_feature, predictor_feature, ignore_features):
    df, categorical_cols = encode_categorical_cols(df)
    # categorical_cols = [col for col in df.columns if df[col].dtype=="O"]
    # categorical_cols = [col.lower() for col in categorical_cols]
    df.columns = df.columns.str.lower()
    current_slopes = build_regression_model(df, response_feature, predictor_feature)
    v1 = [1, 0] if current_slopes[predictor_feature] >= 0 else [0, 1]
    # current_coefs = build_logistic_regression_model(df, response_feature, predictor_feature, categorical_cols)
    # print(f"current: {current_slopes}")
    # print(f"current_coefs: {current_coefs}")
    grouped_slopes_dict = dict()
    for group_feature in set(df.columns) - set(ignore_features):
        if group_feature != response_feature and group_feature != predictor_feature and len(df[group_feature].unique()) < groups_num_threshold:
            slopes = []
            data_groups, _ = get_groups(df, group_feature)
            for data_group in data_groups:
                slopes.append(build_regression_model(data_group, response_feature, predictor_feature))
                # sns.regplot(x=predictor_feature, y=response_feature, data=data_group)
            # grouped_slopes = df.groupby(group_feature).apply(build_regression_model, response_feature, predictor_feature)
            # grouped_coefs = df.groupby(group_feature).apply(build_logistic_regression_model, response_feature, predictor_feature, categorical_cols)
            # print(f"slopes: {slopes}")
            # print(f"grouped_slopes: {grouped_coefs}")
            grouped_slopes_dict[group_feature] = slopes
    
    # build vector representations and check if biased
    biased_degrees_dict = dict()
    unbiased_degrees_dict = dict()
    for group_feature, slopes in grouped_slopes_dict.items():
        v2 = build_reg_represent_vector(slopes, predictor_feature)
        is_biased, bias_degree = check_if_biased(v1, v2)
        if is_biased:
            biased_degrees_dict[group_feature] = bias_degree
        else:
            unbiased_degrees_dict[group_feature] = bias_degree
    # sns.regplot(x=predictor_feature, y=response_feature, data=df)
    print(biased_degrees_dict)
    return biased_degrees_dict, unbiased_degrees_dict


# plot the regression results
def plot_reg_results(df, response_feature, predictor_feature, group_feature):
    plt.figure()
    sns.regplot(x=predictor_feature, y=response_feature, data=df)
    plt.title('Regression on the Entire Data', fontsize=16, fontweight='bold')
    plt.xlabel(f'{predictor_feature}', fontsize=14)
    plt.ylabel(f'{response_feature}', fontsize=14)
    plt.figure()
    data_groups, _ = get_groups(df, group_feature)
    for data_group in data_groups:
        sns.regplot(x=predictor_feature, y=response_feature, data=data_group)
        plt.title(f'Regression on Groups of {group_feature}', fontsize=16, fontweight='bold')
        plt.xlabel(f'{predictor_feature}', fontsize=14)
        plt.ylabel(f'{response_feature}', fontsize=14)



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
    elif option == '3':
        query = input('Query: ')
        # query = 'SELECT avg(admit) from admissions_data GROUP BY Gender'
        analyse_dataset(query, mysql_engine)
    elif option == '4':
        query = input('Query: ')
        # query = 'SELECT avg(admit) from admissions_data GROUP BY Gender'
        ignore_features = []
        print(process_query_regression(query, mysql_engine, ignore_features))
    # check_for_bias(df, 'Treated', 'Treated', 'Survived')
    # check_for_avg_reverse(df, 'Gender', 'Dept', 'Admit')
    # check_for_avg_reverse(df, 'player', 'year', 'outcome')
    plt.show()



if __name__ == '__main__':
    main()