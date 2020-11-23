import numpy as np
import pandas as pd

def get_numeric_columns(df):
    """
    # Returns a list of numeric columns of the dataframe df

    # Parameters:
    #    df (Pandas dataframe): The dataframe from which extract the columns 

    # Returns:
    #    A list of columns names (strings) corresponding to numeric columns

    """
    numeric_columns = list(df._get_numeric_data().columns)
    return numeric_columns

def get_unique_df_values(df, dtypes):
    """
    # Returns a dictionary of type {dict: list}, corresponding to {columns -> unique values}

    # Parameters:
    #    df (Pandas dataframe): The dataframe from which extract the columns unique values
    #   dtypes (dictionary): Dictionary of column types
    
    # Returns:
    #    A dictionary of {columns -> unique values}

    """

    unique_values = {}

    for i, col in enumerate(df.columns): # Not numeric column

        if (dtypes[col] == 'object'): # If categorical column

            # Unique values
            unique_values[col] = df[col].unique()
    
    return unique_values

def split(dataset, y):
    """
    # Returns a shuffled, stratified split of the input dataset

    # Parameters:
    #   dataset (Pandas dataframe): The dataframe from which extract the columns unique values
    #   y (string): name of the column representing the target, i.e. column on which stratify

    # Returns:
    #    A dictionary of {columns -> unique values}

    """
    # Shuffle data
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    ochenta, veinte = train_test_split(dataset, test_size = 0.2, random_state = 0, stratify = dataset[y])

    # 45% Black-Box data, 35% Attack data
    bb, att = train_test_split(ochenta, test_size = 0.45, random_state = 0, stratify = ochenta[y])

    # 85% Training Black-Box, 15% Validation Black-Box
    bb_train, bb_val = train_test_split(bb, test_size = 0.15, random_state = 0, stratify = bb[y])
    
    # 85% Training Shadow models, 15% Validation Shadow models
    sh_train, sh_val = train_test_split(att, test_size = 0.15, random_state = 0, stratify = att[y])

    # 10% records to explain, 10% test set
    r2E, test = train_test_split(veinte, test_size = 0.5, random_state = 0, stratify = veinte[y])

    return bb_train, bb_val, sh_train, sh_val, r2E, test
