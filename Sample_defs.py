# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:16:48 2024

@author: kenneyke
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

def calculate_normalized_class_weights(y_train, class_label_weights):
    # Map the predefined weights to the classes found in y_train
    class_weights = {}
    total_weight = 0
    
    # Create a mapping from class labels to weights
    weight_mapping = dict(zip(class_label_weights.iloc[:, 0], class_label_weights.iloc[:, 1]))
    
    # Assign weights based on y_train distribution and the predefined weights
    for class_label in y_train.unique():
        class_weights[class_label] = weight_mapping.get(class_label, 1)
        total_weight += class_weights[class_label]
    
    # Normalize the weights so they sum to 1
    class_weights = {k: v / total_weight for k, v in class_weights.items()}
    
    return class_weights


def random_sample_split(csv_path, test_size=0.2, random_state=None):
    """
    random_state (int or None): Controls the random state for reproducibility. 
    Use the same integer for the same sample across multiple runs or None for different samples each time.
    """
    # Load the data from the CSV file
    data = pd.read_csv(csv_path)

    # Calculate the number of test samples
    test_count = int(len(data) * test_size)

    # Randomly sample test data from the original dataset
    test_data = data.sample(n=test_count, random_state=random_state)

    # Drop the test data samples from the original dataset to create the training data
    train_data = data.drop(test_data.index)

    return train_data, test_data


def flexible_sample_split(csv_path, test_size=0.2, random_state=None, train_random_state=None):
    """
        random_state (int or None): Controls the overall random state for reproducibility.
        train_random_state (int or None): Controls the random state for training data.
    """
    # Load the data from the CSV file
    data = pd.read_csv(csv_path)

    # Determine random states based on the parameters
    final_test_random_state = random_state
    final_train_random_state = train_random_state if train_random_state is not None else random_state

    # Split the data into training and testing sets
    train_data, Test20 = train_test_split(data, test_size=test_size, random_state=final_test_random_state)

    # Shuffle the training data to introduce randomness
    np.random.seed(final_train_random_state)  # Set the random seed
    np.random.shuffle(train_data.values)

    # Define the proportions for training splits
    train_splits = [0.2, 0.4, 0.6, 0.8]
    training_sets = []

    # Calculate and store each training split
    for split in train_splits:
        subset_size = int(len(train_data) * split/0.8)
        train_subset = train_data.iloc[:subset_size]  # Select the first subset_size rows
        training_sets.append(train_subset)

    # Unpack the list to individual variables
    Train20, Train40, Train60, Train80 = training_sets

    # Return all data splits as separate variables
    return Train20, Train40, Train60, Train80, Test20, train_data

# Example usage:   
csvDataPath=r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\6_Analysis\1HWY_Segment-Segment\1_T01_HWY068_SB_20200603_173632 Profiler.zfs_18_manual_label_hi_segVono_data.csv'
Train_data, Test_data = random_sample_split(csvDataPath, test_size=0.2, random_state=0)

Train20, Train40, Train60, Train80, Test20, train_data = flexible_sample_split(csvDataPath, random_state=42, train_random_state=42)

# # Training
# Train_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class', 'Ext_class', 'bb_centerX', 'bb_centerY', 'bb_centerZ']   #Columns to exclude from the X data
# X_train = Train_data.drop(columns=Train_cols_to_remove, axis=1)  
# y_train = Train_data['Ext_class']

# label_Weight_path = r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\6_Analysis\Label_Class_Weights.xlsx'
# class_label_weights = pd.read_excel(label_Weight_path)
# class_weights = calculate_normalized_class_weights(y_train, class_label_weights)