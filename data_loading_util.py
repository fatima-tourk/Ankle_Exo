import numpy as np
import os
import pandas as pd

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import time
from tensorflow.keras.callbacks import CSVLogger
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#########################################################################################
# LOADING FILES

# Unilateral
def get_files_to_use(root_folder, subject_nums, sides, trial_nums):
    subject_strings = convert_subject_nums_to_strings(subject_nums)
    trial_strings = convert_trial_nums_to_strings(trial_nums)
    files_to_use = []
    for subject_string in subject_strings:
        subject_folder = os.path.join(root_folder, subject_string)
        filenames = os.listdir(subject_folder)
        for side in sides:
            for trial_string in trial_strings:
                for f in filenames:
                    if side in f and trial_string in f:
                        files_to_use.append(os.path.join(subject_folder, f))
    return(files_to_use)

# Bilateral
def get_separate_left_right_files_to_use(root_folder, subject_nums, trial_nums):
    subject_strings = convert_subject_nums_to_strings(subject_nums)
    trial_strings = convert_trial_nums_to_strings(trial_nums)
    
    left_files = []
    right_files = []
    
    for subject_string in subject_strings:
        subject_folder = os.path.join(root_folder, subject_string)
        filenames = os.listdir(subject_folder)
        
        for trial_string in trial_strings:
            for f in filenames:
                if trial_string in f:
                    if "_LEFT.csv" in f:
                        left_files.append(os.path.join(subject_folder, f))
                    elif "_RIGHT.csv" in f:
                        right_files.append(os.path.join(subject_folder, f))
    
    return left_files, right_files


# Prepping files to be loaded
def convert_trial_nums_to_strings(list_of_trial_nums: int):
    mystrings = []
    for num in list_of_trial_nums:
        if num < 10:
            mystrings.append('T0' + str(num))
        else:
            mystrings.append('T' + str(num))
    return mystrings


def convert_subject_nums_to_strings(list_of_subject_nums: int):
    mystrings = []
    for num in list_of_subject_nums:
        if num < 10:
            mystrings.append('S0' + str(num))
        else:
            mystrings.append('S' + str(num))
    return mystrings

#########################################################################################

# LOSS FUNCTIONS
def custom_loss(y_actual, y_pred):
    mask = kb.greater_equal(y_actual, 0)
    mask = tf.cast(mask, tf.float32)
    custom_loss = tf.math.reduce_sum(
        kb.square(mask*(y_actual-y_pred)))/tf.math.reduce_sum(mask)
    return custom_loss


def custom_loss_for_ramp(y_actual, y_pred):
    # Create a mask where `y_actual` is not equal to -100
    mask = kb.not_equal(y_actual, -100)
    # Cast the mask to float32
    mask = tf.cast(mask, tf.float32)
    # Use the mask to ignore the values where `y_actual` is -100
    masked_squared_error = kb.square(mask * (y_actual - y_pred))
    # Calculate the sum of the squared errors where the mask is True
    numerator = tf.math.reduce_sum(masked_squared_error)
    # Calculate the sum of the mask values (essentially the count of non-masked elements)
    denominator = tf.math.reduce_sum(mask)
    # To avoid division by zero, add a small constant to the denominator
    denominator = tf.where(tf.equal(denominator, 0), tf.constant(1, dtype=tf.float32), denominator)
    # Compute the mean squared error while ignoring the masked values
    custom_loss_value = numerator / denominator
    return custom_loss_value

##########################################################################################

# LOAD AND MERGE FILES

# Unilateral
def load_file(myfile):
    stance_phase_to_get = 'TM_Stance_Phase'
    df = pd.read_csv(myfile, usecols=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y',
                     'gyro_z', 'ankle_angle', 'ankle_velocity', 'Ramp', 'Velocity', 'TM_Is_Stance_Phase', stance_phase_to_get])
    return df.values

# Bilateral
def load_file_bilateral(myfile, ipsilateral_is_right=True):
    stance_phase_to_get = 'TM_Stance_Phase'
    df = pd.read_csv(myfile, usecols=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y',
                     'gyro_z', 'ankle_angle', 'ankle_velocity', 'Ramp', 'Velocity', 'TM_Is_Stance_Phase', stance_phase_to_get])
    
    # Extract the side information from the file name based on the suffix
    file_name = myfile.split('/')[-1]  # Extract just the file name from the path
    if file_name.endswith('_LEFT.csv'):
        side = 'LEFT'
    elif file_name.endswith('_RIGHT.csv'):
        side = 'RIGHT'
    else:
        raise ValueError("File name does not have a valid side suffix (_LEFT.csv or _RIGHT.csv)")

    # Determine the side_suffix based on ipsilateral_is_right
    if ipsilateral_is_right:
        side_suffix = "_ipsi" if side == "RIGHT" else "_contra"
    else:
        side_suffix = "_contra" if side == "RIGHT" else "_ipsi"

    side_suffix2 = "_RIGHT" if side == "RIGHT" else "_LEFT"

    df = df.add_suffix(side_suffix)
    df = df.add_suffix(side_suffix2)
    
    return df

# Merge bilateral
def merge_files_with_lists(ipsilateral_files, contralateral_files, ipsilateral_is_right=True):
    # Make sure the ipsilateral and contralateral file lists have the same length
    if len(ipsilateral_files) != len(contralateral_files):
        raise ValueError("Ipsilateral and contralateral file lists must have the same length.")

    combined_data_list = []  # List to store the merged DataFrames

    for ipsilateral_file, contralateral_file in zip(ipsilateral_files, contralateral_files):
        # Load data from ipsilateral and contralateral files
        #ipsilateral_data = load_file_bilateral(ipsilateral_file, side="RIGHT" if ipsilateral_is_right else "LEFT", ipsilateral_is_right=ipsilateral_is_right)
        #contralateral_data = load_file_bilateral(contralateral_file, side="LEFT" if ipsilateral_is_right else "RIGHT", ipsilateral_is_right=ipsilateral_is_right)

        ipsilateral_data = load_file_bilateral(ipsilateral_file, ipsilateral_is_right=ipsilateral_is_right)
        contralateral_data = load_file_bilateral(contralateral_file, ipsilateral_is_right=ipsilateral_is_right)
        #print("ipsi: ", ipsilateral_data.columns)
        #print("contra: ", contralateral_data.columns)

        # Drop the contralateral "Ramp_right" and "Velocity_right" columns
        if ipsilateral_is_right == True:
            contralateral_data = contralateral_data.drop(columns=['Ramp_contra_LEFT', 'Velocity_contra_LEFT'])
        else:
            contralateral_data = contralateral_data.drop(columns=['Ramp_contra_RIGHT', 'Velocity_contra_RIGHT'])

        # Assign a common index (assuming both DataFrames have the same number of rows)
        ipsilateral_data.index = range(len(ipsilateral_data))
        contralateral_data.index = range(len(contralateral_data))

        # Merge the two DataFrames using index-based merge
        combined_data = pd.concat([ipsilateral_data, contralateral_data], axis=1)  # Combine them side by side
        combined_data_list.append(combined_data)

    # Combine all merged DataFrames into a single DataFrame
    combined_data = pd.concat(combined_data_list, ignore_index=True)
    #print(combined_data.columns)

    return combined_data.values

###########################################################################################################################################
# MANIPULATE COLUMNS
def replace_column_values_simple(data, column_index, original_value, replacement_value):
    """
    Replace values in a specific column of a NumPy array.

    Parameters:
    - data (numpy.ndarray): The input array.
    - column_index (int): Index of the column to be replaced.
    - original_value: The value to be replaced.
    - replacement_value: The value to replace the original_value with.

    Returns:
    - numpy.ndarray: The modified array.
    """
    col_data = data[:, column_index]
    col_data[col_data == original_value] = replacement_value
    data[:, column_index] = col_data
    return data

def manipulate_ss_col(ss_col):
    flag = 0  # Flags: 0 -> looking for first positive, 1 -> searching for 1, 2 -> holding 1, 3 -> setting to -1, 4 -> holding 0
    counter = 0
    first_positive_found = False  # Indicator for the first positive value

    for i in range(len(ss_col)):
        if not first_positive_found:
            if ss_col[i] > 0:
                first_positive_found = True
                flag = 1  # Start looking for the next 1
            else:
                ss_col[i] = -1  # Set to -1 until the first positive value is found
        else:
            if flag == 1 and ss_col[i] == 1:
                # When 1 is found, start holding at 1
                flag = 2
                counter = 1
            elif flag == 2:
                # Hold 1 for 15 points
                if counter < 15:
                    ss_col[i] = 1
                    counter += 1
                else:
                    # Then go to -1 for the next 10 points
                    flag = 3
                    counter = 1
                    ss_col[i] = -1
            elif flag == 3:
                # Hold -1 for 10 points
                if counter < 10:
                    ss_col[i] = -1
                    counter += 1
                else:
                    # Then go back to 0 and start looking for 1 again
                    flag = 4
                    counter = 0
            elif flag == 4 and ss_col[i] != 0:
                # As soon as the value starts increasing, start looking for 1 again
                flag = 1
    
    return ss_col

#############################################################################################
# SHUFFLE, SPLIT, AND LABEL DATA

def split_data(data, labels, split_fraction=0.8):
    """
    Split the data and labels into training and validation sets.

    Parameters:
    - data (numpy.ndarray): Input data.
    - labels (list of numpy.ndarray): List of output data arrays.
    - split_fraction (float): Fraction of data to be used for training (default is 0.8).

    Returns:
    - tuple: Tuple containing training and validation data and labels.
    """
    split_num = int(np.rint(split_fraction * data.shape[0]))

    # Split data
    x_train = data[:split_num, :, :]
    x_valid = data[split_num:, :, :]

    # Split labels
    labels_train = [label[:split_num, :] for label in labels]
    labels_valid = [label[split_num:, :] for label in labels]

    return x_train, x_valid, labels_train, labels_valid


# Bilateral
def shuffle_and_extract_features_labels(data, window_size):
    """
    Shuffle the data and extract features and labels.

    Parameters:
    - data (numpy.ndarray): Input data.
    - window_size (int): Size of the window for label extraction.

    Returns:
    - tuple: Tuple containing shuffled data, features (x), and labels (y).
    """
    # Shuffle the data along axis=0
    shuffled_data = tf.random.shuffle(data)

    # Extract features and labels
    x = tf.concat([shuffled_data[:, :, :8], shuffled_data[:, :, 12:20]], axis=-1)

    y_v_i = shuffled_data[:, window_size - 1:, -11]
    y_r_i = shuffled_data[:, window_size - 1:, -12]
    y_sp_i = shuffled_data[:, window_size - 1:, -14]
    y_ss_i = shuffled_data[:, window_size - 1:, -13]
    y_sp_c = shuffled_data[:, window_size - 1:, -2]
    y_ss_c = shuffled_data[:, window_size - 1:, -1]

    return shuffled_data, x, y_v_i, y_r_i, y_sp_i, y_ss_i, y_sp_c, y_ss_c

# Unilateral
def shuffle_and_extract_features_labels_uni(data, window_size):
    """
    Shuffle the data and extract features and labels.

    Parameters:
    - data (numpy.ndarray): Input data.
    - window_size (int): Size of the window for label extraction.

    Returns:
    - tuple: Tuple containing shuffled data, features (x), and labels (y).
    """
    shuffled_data = tf.random.shuffle(data) 
    num_channels = 8
    x = shuffled_data[:, :, :num_channels]
    y_v = shuffled_data[:, window_size-1:,-1]
    y_r = shuffled_data[:, window_size-1:,-2]
    y_sp = shuffled_data[:, window_size-1:,-4]
    y_ss = shuffled_data[:, window_size-1:,-3]

    return shuffled_data, x, y_v, y_r, y_sp, y_ss

########################################################################################


class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)