
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
            
            
    def correct_query(self, query) -> str:
        """Detect and correct the biased query if Simpson's paradox is present in results

        Args:
            query ([str]): Given query string

        Returns:
            [str]: Corrected unbiased query
        """
        agg_attributes, tables, common_attributes, columns = self.extract_features(query)
        agg_attributes = [att.replace('`', '') for att in agg_attributes]
        tables = [table.replace('`', '') for table in tables]
        common_attributes = [att.replace('`', '') for att in common_attributes]
        columns = [att.replace('`', '') for att in columns]
        if len(agg_attributes) > 0 and len(common_attributes) > 0:
            return detect_and_correct_query_bias(query, sql_engine, agg_attributes, tables, common_attributes, columns)
        else:
            return query
        
        
    def extract_features(self, query) -> tuple(list, list, list, list):
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