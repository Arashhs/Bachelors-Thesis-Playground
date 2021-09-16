
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

import statsmodels.api as sm
import seaborn as sns
import sqlparse

# Parameters
groups_num_threshold = 100


class query_bias_correction:
    """This class is used to detect and correct query biases
    
    Attributes:
        sql_engine: An SQLAlchemy Engine instance 
    """
    def __init__(self, sql_engine) -> None:
        self.sql_engine = sql_engine
        

    def init_sql_db(self, dataset_directory) -> None:
        """Create SQL tables from CSV datasets which are located in the given directory

        Args:
            dataset_directory ([str]): Directory location where the CSV datasets are stored in
        """
        # get datasets' names
        csv_files = [os.path.join(dataset_directory, file_name) for file_name in os.listdir(dataset_directory)]
        dataset_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in csv_files]
        for i, dataset_name in enumerate(dataset_names):
            # read the CSV dataset and make the corresponding table in the database
            df = pd.read_csv(csv_files[i])
            df.to_sql(dataset_name, con=self.sql_engine, index=False, if_exists='replace')
            
            
    def process_query(self, query, ignore_features=[], plot_results=True) -> str:
        """Detect and correct the biased query if Simpson's paradox is present in results

        Args:
            query ([str]): Given query string
            plot_results([bool]): Whether or not to plot the results. Optional, default to True.

        Returns:
            [str]: Corrected unbiased query
        """
        agg_attributes, tables, common_attributes, columns = self.extract_features(query)
        agg_attributes = [att.replace('`', '') for att in agg_attributes]
        tables = [table.replace('`', '') for table in tables]
        common_attributes = [att.replace('`', '') for att in common_attributes]
        columns = [att.replace('`', '') for att in columns]
        if len(agg_attributes) > 0 and len(common_attributes) > 0:
            return self.detect_and_correct_query_bias(query, agg_attributes, common_attributes, ignore_features, plot_results)
        else:
            return query
        
        
    def extract_features(self, query) -> 'tuple[list, list, list, list]':
        """Extract [Aggregated, Common] Attributes and the Target Tables and Columns from query

        Args:
            query ([str]): Given query string

        Returns:
            (List(str), List(str), List(str), List(str)): Aggregated Attributes, Common Attributes, 
            Target Tables, Target Columns
        """
        select_seen, from_seen, group_by_seen = (False, False, False)
        agg_attributes, common_attributes, tables, columns = ([], [], [], [])
        parsed_query = sqlparse.parse(query)[0]
        for token in parsed_query:
            if select_seen:
                if isinstance(token, sqlparse.sql.IdentifierList):
                    for identifier in token.get_identifiers():
                        if isinstance(identifier, sqlparse.sql.Function):
                            agg_attributes += self.get_agg_from_token(identifier)
                            continue
                        columns.append(identifier.value.lower())
                        # print("{} {}\n".format("Attr = ", identifier) )
                elif isinstance(token, sqlparse.sql.Identifier):
                    columns.append(token.value.lower())
                    # print("{} {}\n".format("Attr = ", token))
                elif isinstance(token, sqlparse.sql.Function):
                    agg_attributes += self.get_agg_from_token(token)
            if from_seen:
                if isinstance(token, sqlparse.sql.IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.append(identifier.value.lower())
                        # print("{} {}\n".format("TAB = ", identifier) )
                elif isinstance(token, sqlparse.sql.Identifier):
                    tables.append(token.value.lower())
                    # print("{} {}\n".format("TAB = ", token))
            if group_by_seen:
                if isinstance(token, sqlparse.sql.IdentifierList):
                    for identifier in token.get_identifiers():
                        common_attributes.append(identifier.value.lower())
                        # print("{} {}\n".format("GROUPBY att = ", identifier) )
                elif isinstance(token, sqlparse.sql.Identifier):
                    common_attributes.append(token.value.lower())
                    # print("{} {}\n".format("GROUPBY att = ", token))
                
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
            
        return agg_attributes, tables, common_attributes, columns
    
    
    def get_agg_from_token(self, token) -> list:
        """Extract the aggregated attribute from the given token

        Args:
            token ([sqlparse.sql.Function]): The given sqlparse.sql.Function token
            agg_attributes ([list]): list of current aggregated attributes

        Returns:
            [list]: list of extracted aggregated attributes
        """
        agg_attributes = []
        avg_seen = False
        for sub_token in token.tokens:
            if avg_seen:
                for par_token in sub_token.tokens:
                    if isinstance(par_token, sqlparse.sql.IdentifierList):
                        for identifier in par_token.get_identifiers():
                            # print("{} {}\n".format("TargetAtt = ", identifier) )
                            agg_attributes.append(identifier.value.lower())
                    elif isinstance(par_token, sqlparse.sql.Identifier):
                        agg_attributes.append(par_token.value.lower())
                        # print("{} {}\n".format("TargetAtt = ", par_token) )
                avg_seen = False
            if sub_token.value.upper() == "AVG":
                avg_seen = True
        return agg_attributes

    
    # Check whether Simpson's paradox is present and correct it if present
    def detect_and_correct_query_bias(self, query, agg_attributes, common_attributes, ignore_features, plot_results) -> str:
        """Check whether Simpson's paradox is present and correct it if present

        Args:
            query ([str]): Input query String
            agg_attributes ([list]): List of aggregated attributes
            common_attributes ([list]): List of common attributes 

        Returns:
            [str]: Corrected unbiased query
        """
        df = pd.read_sql(self.preprocess_query(query), self.sql_engine)
        df.columns = df.columns.str.lower()
        biased_degrees_dict = dict()
        unbiased_degrees_dict = dict()
        for test_attribute in df.columns:
            if test_attribute not in common_attributes and test_attribute not in agg_attributes and len(df[test_attribute].unique()) < groups_num_threshold and test_attribute not in ignore_features:
                # is_query_biased, bias_degree = self.check_for_bias(df, common_attributes[0], test_attribute, agg_attributes[0])
                                
                first_groups, first_att_vals = self.get_groups(df, [common_attributes[0]])
                second_groups, second_att_vals  = self.get_groups(df, [common_attributes[0], test_attribute])
                first_avgs = self.get_groups_avgs(first_groups, agg_attributes[0])
                second_avgs = self.get_groups_avgs(second_groups, agg_attributes[0])
                first_group_avgs = {key:value for key, value in zip(first_att_vals, first_avgs)}
                second_group_avgs = {key:value for key, value in zip(second_att_vals, second_avgs)}
                is_query_biased, bias_degree = self.is_biased(first_group_avgs, second_group_avgs)

                # print(test_attribute, is_query_biased, bias_degree)
                if is_query_biased:
                    biased_degrees_dict[test_attribute] = bias_degree
                else:
                    unbiased_degrees_dict[test_attribute] = bias_degree
        if len(biased_degrees_dict) > 0:
            # Query was biased
            # Select the attribute for which the bias degree was highest
            best_test_att, highest_bias_degree = max(biased_degrees_dict.items(), key=lambda v: v[1])
            corrected_query = self.correct_query(query, best_test_att)
            if plot_results:
                self.plot_query_results(query, corrected_query, common_attributes[0], best_test_att, agg_attributes[0])
            return corrected_query
        else:
            # Query was unbiased
            return query


    # Only hold essential parts of query for retreiving dataframe
    def preprocess_query(self, query) -> str:
        """Only hold essential parts of query for retreiving all of the target data as dataframe

        Args:
            query ([str]): Input query string

        Returns:
            [str]: Minimal query string that retrieves all of the target data
        """
        unwanted_keywords = ['SELECT', 'GROUP', 'ORDER', 'HAVING']
        seen_unwanted = False
        formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
        sql_lines = formatted_query.split('\n')
        manipulated_sql = 'SELECT *'
        for line in sql_lines:
            if line.startswith((' ', '\t')):
                # print(f'param-line: {line}')
                # print(f"parameter: {line.strip().replace(r',', '')}")
                # print()
                pass
            else:
                # print(f'Statement: {line}')
                splitted_stmt = line.split()
                keyword = splitted_stmt[0]
                if keyword.upper() in unwanted_keywords:
                    seen_unwanted = True
                else:
                    seen_unwanted  = False
                # print(f'Keyword: {splitted_stmt[0]}')
                # print(f"Param: {splitted_stmt[1].strip().replace(r',', '')}")
                # print()
            
            if not seen_unwanted:
                manipulated_sql += '\n' + line
        # print(f'manipulated_sql: {manipulated_sql}')
        return manipulated_sql


    # get the desired groups based on common attributes in dataset
    def get_groups(self, dataframe, common_att_list) -> 'tuple[list, list]':
        """Get the desired groups based on the selected attributes

        Args:
            dataframe ([pandas.DataFrame]): The given dataframe
            common_att_list ([list]): List of attributes on which we want to group data

        Returns:
            [tuple(list, list)]: List of groups of data as well as the unique values on the given attribute
        """
        splited_df = dataframe.groupby(common_att_list)
        common_att_vals = list(splited_df.groups.keys())
        groups = []
        for att_vals in common_att_vals:
            groups.append(splited_df.get_group(att_vals))
        return groups, common_att_vals


    # get the mean of the desired attribute for several groups
    def get_groups_avgs(self, groups, agg_attribute) -> list:
        """Get the mean of the desired attribute for the given groups

        Args:
            groups ([list]): List of groups of data 
            agg_attribute ([string]): name of the 

        Returns:
            [list]: List of the average of target attribute for the given groups 
        """
        avgs = [group[agg_attribute].mean() for group in groups]
        return avgs


    # Check if the averages of target attribute reverse if we disaggregate data using test_attribute
    def check_for_bias(self, dataframe, common_attribute, test_attribute, agg_attribute) -> 'tuple[bool, float]':
        """Check if the averages of target attribute reverses if we disaggregate data using test_attribute

        Args:
            dataframe ([pandas.DataFrame]): Dataframe on which we do analysis
            common_attribute ([type]): Attribute on which we group the data
            test_attribute ([type]): The test attribute which we use for disaggregating data
            agg_attribute ([type]): The attribute which average is important to us

        Returns:
            [tuple(bool, float)]: Whether or not the configuration is biased (Simpson is present) and the bias degree
        """
        first_groups, first_att_vals = self.get_groups(dataframe, [common_attribute])
        second_groups, second_att_vals  = self.get_groups(dataframe, [common_attribute, test_attribute])
        first_avgs = self.get_groups_avgs(first_groups, agg_attribute)
        second_avgs = self.get_groups_avgs(second_groups, agg_attribute)
        first_group_avgs = {key:value for key, value in zip(first_att_vals, first_avgs)}
        second_group_avgs = {key:value for key, value in zip(second_att_vals, second_avgs)}
        return self.is_biased(first_group_avgs, second_group_avgs)
    
    
    # check if the Simpson's paradox has happened based on two set of groups
    def is_biased(self, first_gp_avgs, second_gp_avgs) -> 'tuple[bool, float]':
        """Check if the Simpson's paradox is detected after disaggregating data using the test attribute

        Args:
            first_gp_avgs ([dict(str, float)]): dictionary containing the first group avgs (before disaggregation)
            second_gp_avgs ([dict(str, float)]): dictionary containing the second group avgs (after disaggregation)

        Returns:
            [tuple(bool, float)]: Whether or not the configuration is biased (Simpson is present) and the bias degree
        """
        common_att_vals = list(first_gp_avgs.keys())
        test_attribute_vals = np.unique([val[1] for val in second_gp_avgs.keys()])
        second_gp_avgs = self.augment_zeroes(second_gp_avgs, common_att_vals, test_attribute_vals)
        agg_bigger_num = []
        test_bigger_num = []
        for common_att_val in common_att_vals:
            agg_bigger_num.append((np.sum([first_gp_avgs[common_att_val] >= first_gp_avgs[val] for val in common_att_vals]) - 1))
        test_bigger_num = 0 * np.array(agg_bigger_num)
        for test_att_val in test_attribute_vals:
            current_vec = []
            for common_att_val in common_att_vals:
                current_vec.append((np.sum([second_gp_avgs[common_att_val, test_att_val] >= second_gp_avgs[val, test_att_val] for val in common_att_vals]) - 1))
            test_bigger_num = np.add(test_bigger_num, current_vec)
        test_bigger_num = list(test_bigger_num)
        # bias_degree = get_bias_degree(agg_bigger_num, test_bigger_num)
        # if np.all(np.array(test_bigger_num) == 0):
        #     bias_degree = 0
        # if test_bigget_num is all zeroes or the bias degree is lower than the threshold, return False
        # if bias_degree < bias_degree_threshold:
        #     return False, bias_degree
        # return True, bias_degree
        is_biased, bias_degree = self.check_if_biased(agg_bigger_num, test_bigger_num)
        return is_biased, bias_degree
    
    
    # add key-pairs with the the value 0 if key-pair does not exist in dict
    def augment_zeroes(self, group_avgs, common_att_vals, test_vals) -> 'dict[str, float]':
        """Add key-pairs with the the value 0 if key-pair does not exist in dict

        Args:
            group_avgs ([dict(str, float)]): Dictionary containing the averages of the agg_attribute for each group
            common_att_vals ([list]): The unique values of the common attribute
            test_vals ([list]): The unique pairs of the values of (common, test) attribute

        Returns:
            [dict(str, float)]: Dictionary containing the averages after augmenting zeroes
        """
        for test_val in test_vals:
            for common_att_val in common_att_vals:
                if (common_att_val, test_val) not in group_avgs:
                    group_avgs[common_att_val, test_val] = 0
        return group_avgs


    # Check if the representations v1 and v2 indicate biased result
    def check_if_biased(self, v1, v2) -> 'tuple[bool, float]':
        """Check if the representations v1 and v2 indicate biased result

        Args:
            v1 ([list]): V1 representation showing the number of values bigger that the other list values at each index for the aggregated data
            v2 ([list]): V2 representation showing the number of values bigger that the other list values at each index for the disaggregated data

        Returns:
            [tuple(bool, float)]: Whether or not the configuration is biased (Simpson is present) and the bias degree
        """
        ind_1 = v1.index(max(v1))
        ind_2 = v2.index(max(v2))
        bias_degree = self.get_bias_degree(v1, v2)
        if ind_1 != ind_2 and not np.all(np.array(v2) == 0):
            return True, bias_degree
        return False, bias_degree
    
    
    # get the bias degree
    def get_bias_degree(self, vec1, vec2) -> float:
        """Calculate and return the bias degree given vec1 and vec2

        Args:
            vec1 ([list]): V1 representation showing the number of values bigger that the other list values at each index for the aggregated data
            vec2 ([list]): V2 representation showing the number of values bigger that the other list values at each index for the disaggregated data

        Returns:
            [float]: The degree of bias given vec1 and vec2
        """
        return 1 - self.get_cosine_score(vec1, vec2)
    
    
    # getting normalized of a vector
    def get_normalized(self, vector) -> list:
        """Get the normalized of the given vector

        Args:
            vector ([list]): A vector

        Returns:
            [list]: The normalized version of the vector
        """
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if norm == 0: 
            return vector
        return vector / norm


    # calculating cosine score between two vectors
    def get_cosine_score(self, vec1, vec2) -> list:
        """Calculate and return the cosine score between two vectors

        Args:
            vec1 ([list]): First vector
            vec2 ([list]): Second vector

        Returns:
            [float]: Cosine similarity score between vec1 and vec2
        """
        v1 = self.get_normalized(vec1)
        v2 = self.get_normalized(vec2)
        cosine_sim = np.dot(v1, v2)
        return cosine_sim
    
    
    # correct the biased query by disaggregating with a candidate attribute
    def correct_query(self, query, disaggregation_att) -> str:
        """correct the biased query by disaggregating by a candidate attribute

        Args:
            query ([str]): Given query string
            disaggregation_att ([str]): Name of the chosen attribute for disaggregation 

        Returns:
            [str]: The corrected query after adding the disaggregation_attribute to the GROUP BY attributes
        """
        target_keywords = ['SELECT', 'GROUP']
        seen_target = False
        formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
        sql_lines = formatted_query.split('\n')
        corrected_query = ''
        for line in sql_lines:
            if line.startswith((' ', '\t')):
                # print(f'param-line: {line}')
                # print(f"parameter: {line.strip().replace(r',', '')}")
                # print()
                pass
            else:
                if seen_target:
                    corrected_query += f', `{disaggregation_att}`'
                # print(f'Statement: {line}')
                splitted_stmt = line.split()
                keyword = splitted_stmt[0]
                if keyword.upper() in target_keywords:
                    seen_target = True
                else:
                    seen_target  = False
                # print(f'Keyword: {splitted_stmt[0]}')
                # print(f"Param: {splitted_stmt[1].strip().replace(r',', '')}")
                # print()
            corrected_query += '\n' + line
        if seen_target:
            corrected_query += f', `{disaggregation_att}`'
        corrected_query = sqlparse.format(corrected_query, reindent=True, keyword_case='upper')
        # print(f'manipulated_sql: {corrected_query}')
        return corrected_query
    
    
    # plot the corrected query results
    def plot_query_results(self, query, corrected_query, common_att, disagg_att, agg_att):
        """plot the given query results before and after correction

        Args:
            query ([str]): query string which is biased
            corrected_query ([str]): query string after removing bias
            common_att ([str]): the attribute on which data was grouped in the biased query
            disagg_att ([str]): the attribute on which we disaggregate data in order to remove the bias of the query
            agg_att ([str]): the attribute that the original query targets its mean
        """
        df_biased = pd.read_sql(query, self.sql_engine)
        df_biased.columns = df_biased.columns.str.lower().str.replace('`', '')
        df_corrected = pd.read_sql(corrected_query, self.sql_engine)
        df_corrected.columns = df_corrected.columns.str.lower().str.replace('`', '')
        # plotting biased query results
        ax = df_biased.groupby([common_att])['avg(%s)' % agg_att].apply(float).plot(kind='bar', rot=1, legend=False, color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        plt.title('Biased Query Results', fontsize=16, fontweight='bold')
        plt.xlabel(f'{common_att.capitalize()}', fontsize=14)
        plt.ylabel(f'AVG({agg_att.capitalize()})', fontsize=14)
        # annotating the chart
        for p in ax.patches:
            # ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
            ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # plotting corrected query results
        ax = df_corrected.groupby([disagg_att, common_att])['avg(%s)' % agg_att].apply(float).unstack().plot(kind='bar', rot=1)
        plt.title('Corrected Query Results', fontsize=16, fontweight='bold')
        plt.xlabel(f'{disagg_att.capitalize()}', fontsize=14)
        plt.ylabel(f'AVG({agg_att.capitalize()})', fontsize=14)
        # annotating the chart
        for p in ax.patches:
            # ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
            ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.show()
        
        
    
    
    ############ THE SECOND METHOD - USING REGRESSION ############
    # process query, detect and correct Simpson's paradox using regression models if present
    def process_query_regression(self, query, ignore_features=[], plot_results=True) -> str:
        """process query, detect and correct Simpson's paradox using regression models if present

        Args:
            query ([str]): input query string
            ignore_features (list, optional): list of features to be ignored for being selected as disaggregation feature. Defaults to [].
            plot_results (bool, optional): Whether or not to plot results. Defaults to True.

        Returns:
            str: Corrected unbiased query
        """
        agg_attributes, tables, common_attributes, columns = self.extract_features(query)
        agg_attributes = [att.replace('`', '') for att in agg_attributes]
        tables = [table.replace('`', '') for table in tables]
        common_attributes = [att.replace('`', '') for att in common_attributes]
        columns = [att.replace('`', '') for att in columns]
        if len(agg_attributes) > 0 and len(common_attributes) > 0:
            return self.detect_and_correct_query_bias_regression(query, agg_attributes, common_attributes, ignore_features, plot_results)
        else:
            return query
        
        
    # Check whether Simpson's paradox is present and correct it using regression models if present
    def detect_and_correct_query_bias_regression(self, query, agg_attributes, common_attributes, ignore_features, plot_results):
        """Check whether Simpson's paradox is present and correct it using regression models if present

        Args:
            query ([str]): input query string
            agg_attributes ([list[str]]): List of aggregated attributes
            common_attributes ([list[str]]) List of common attributes
            ignore_features ([list[str]]): List of features to be ignored for being selected as disaggregation att.
            plot_results ([str]): Whether or not to plot the results

        Returns:
            [str]: Corrected unbiased query
        """
        df = pd.read_sql(self.preprocess_query(query), self.sql_engine)
        biased_degrees_dict, unbiased_degrees_dict = self.build_regression_models(df, agg_attributes[0], common_attributes[0], ignore_features)
        if len(biased_degrees_dict) > 0:
            # Query was biased
            # Select the attribute for which the bias degree was highest
            best_test_att, highest_bias_degree = max(biased_degrees_dict.items(), key=lambda v: v[1])
            corrected_query = self.correct_query(query, best_test_att)
            if plot_results:
                self.plot_query_results(query, corrected_query, common_attributes[0], best_test_att, agg_attributes[0])
                self.plot_reg_results(df, agg_attributes[0], common_attributes[0], best_test_att)
            return corrected_query
        else:
            # Query was unbiased
            return query
        
        
    # build regression models for testing if Simpson is present
    def build_regression_models(self, df, response_feature, predictor_feature, ignore_features):
        """build a regression model using given inputs

        Args:
            df ([pandas.DataFrame]): Dataframe containing the entire target data
            response_feature ([str]): The response feature for building the first regression model
            predictor_feature ([str]): The predictor feature for building the the first regression model
            ignore_features (list, optional): list of features to be ignored for being selected as disaggregation feature. Defaults to [].

        Returns:
            [tuple[dict, dict]]: Tuple containing biased degrees dictionary and the unbiased dictionary
        """
        df, categorical_cols = self.encode_categorical_cols(df)
        # categorical_cols = [col for col in df.columns if df[col].dtype=="O"]
        # categorical_cols = [col.lower() for col in categorical_cols]
        df.columns = df.columns.str.lower()
        current_slopes = self.build_regression_model(df, response_feature, predictor_feature)
        v1 = [1, 0] if current_slopes[predictor_feature] >= 0 else [0, 1]
        # current_coefs = build_logistic_regression_model(df, response_feature, predictor_feature, categorical_cols)
        # print(f"current: {current_slopes}")
        # print(f"current_coefs: {current_coefs}")
        grouped_slopes_dict = dict()
        for group_feature in set(df.columns) - set(ignore_features):
            if group_feature != response_feature and group_feature != predictor_feature and len(df[group_feature].unique()) < groups_num_threshold:
                slopes = []
                data_groups, _ = self.get_groups(df, group_feature)
                for data_group in data_groups:
                    slopes.append(self.build_regression_model(data_group, response_feature, predictor_feature))
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
            v2 = self.build_reg_represent_vector(slopes, predictor_feature)
            is_biased, bias_degree = self.check_if_biased(v1, v2)
            if is_biased:
                biased_degrees_dict[group_feature] = bias_degree
            else:
                unbiased_degrees_dict[group_feature] = bias_degree
        # sns.regplot(x=predictor_feature, y=response_feature, data=df)
        print(biased_degrees_dict)
        return biased_degrees_dict, unbiased_degrees_dict
    
    
    # encode dataset with categorical data
    def encode_categorical_cols(self, df):
        """encode dataset with categorical data

        Args:
            df ([pandas.DataFrame]): Dataframe containing the entire target data

        Returns:
            [tuple[pandas.DataFrame, list[str]]]: Tuple containing the encoded dataframe and the list of categorical names 
        """
        categorical_cols = [col for col in df.columns if df[col].dtype=="O"]
        for cat_col in categorical_cols:
            df[cat_col] = df[cat_col].astype('category').cat.codes
        return df, categorical_cols


    # build a single regression model and return the parameters
    def build_regression_model(self, data, response_feature, predictor_feature):
        """build a single regression model and return the parameters

        Args:
            data ([pandas.DataFrame]): The target dataframe
            response_feature ([str]): The response feature for building the regression model
            predictor_feature ([str]): The predictor feature for building the regression model

        Returns:
            [OLS.params]: The resulting regression model parameters
        """
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
    
    
    # build representation vector from regression model
    def build_reg_represent_vector(self, slopes, predictor_feature):
        """build representation vector from regression model

        Args:
            slopes ([list[float]]): List of slopes for each group
            predictor_feature ([str]): The predictor feature for building the regression model

        Returns:
            [list[float]]: The vector representation of the resulting regression models for groups
        """
        vec = [0, 0]
        for slope in slopes:
            predictor_slope = slope[predictor_feature]
            if predictor_slope >= 0:
                vec[0] += 1
            else:
                vec[1] += 1
        return vec
    
    
    # plot the regression results
    def plot_reg_results(self, df, response_feature, predictor_feature, group_feature):
        """plot the regression results

        Args:
            df ([pandas.DataFrame]): The target dataframe
            response_feature ([str]): The response feature for building the regression model
            predictor_feature ([str]): The predictor feature for building the regression model
            group_feature ([str]): The feature used for disaggregating the data with
        """
        plt.figure()
        sns.regplot(x=predictor_feature, y=response_feature, data=df)
        plt.title('Regression on the Entire Data', fontsize=16, fontweight='bold')
        plt.xlabel(f'{predictor_feature}', fontsize=14)
        plt.ylabel(f'{response_feature}', fontsize=14)
        plt.figure()
        data_groups, _ = self.get_groups(df, group_feature)
        for data_group in data_groups:
            sns.regplot(x=predictor_feature, y=response_feature, data=data_group)
            plt.title(f'Regression on Groups of {group_feature}', fontsize=16, fontweight='bold')
            plt.xlabel(f'{predictor_feature}', fontsize=14)
            plt.ylabel(f'{response_feature}', fontsize=14)
        plt.show()
        
        
    # analyse dataset to see if Simpson's paradox is present
    def analyse_dataset(self, query):
        df = pd.read_sql(self.preprocess_query(query), self.sql_engine)
        df, categorical_cols = self.encode_categorical_cols(df)
        for agg_att in (set(df.columns) - set(categorical_cols)):
            for common_att in df.columns:
                if common_att != agg_att:
                    for test_att in df.columns:
                        if (test_att != agg_att) and (test_att != common_att):
                            biased, score = self.check_for_bias(df, common_att, test_att, agg_att)
                            if biased:
                                print(self.check_for_bias(df, common_att, test_att, agg_att), f"agg:{agg_att}, common:{common_att}, test:{test_att}")