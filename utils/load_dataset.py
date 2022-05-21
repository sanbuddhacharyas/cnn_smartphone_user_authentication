from joblib import load, dump
import pandas as pd
import os

def load_cnn_training_dataset(input_file):
    with open(input_file, 'rb') as f:      #Import rest data
        return load(f)


def import_raw_dataset(path_acc, path_gyro):
    """
    Imports raw accelerometer and gyroscope data from csv files in S3 bucket then, merge them into single pandas dataframe. The data are merged according to the time column.
    
    Parameters:
    -----------
        path_acc: path of accelerometer data in S3 bucket.
        Path_gyro: path of gyroscope data in S3 bucket.
    
    Returns:
    --------
        Pandas dataframe containing merged accelerometer and gyroscope data
    """

    
    data_acc  = pd.read_csv(path_acc,  names=['Time', 'acc_x', 'acc_y', 'acc_z'])
    data_gyro = pd.read_csv(path_gyro, names=['Time', 'gyr_x', 'gyr_y', 'gyr_z'])
    data      = pd.merge(data_acc, data_gyro, on=['Time'])
    data      = data[['Time', 'gyr_x', 'gyr_y', 'gyr_z', 'acc_x', 'acc_y', 'acc_z']]
    return data


# To load the stored filtered data of the users
def pre_fildata_import(path):
    """
    Import data that has been filtered by rest filter model.
    
    Parameters:
    -----------
        path: pre-processed data path of users
    
    Returns:
    --------
        Tuple containing dataframe and True if loading data is successful else (None, False).
    """

    try:
        with open(path+"/data_fil.csv") as f:
            data_fil = load(f)
        return data_fil, True
    except:
        return None, False


def save_data(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        dump(data, f)


def load_data(save_path):
    with open(save_path, 'rb') as f:
        return load(f)

    