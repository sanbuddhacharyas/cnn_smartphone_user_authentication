import numpy as np
import datetime
import os
from joblib import load, dump
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '../')
from utils.load_dataset import import_raw_dataset, pre_fildata_import, save_data
from utils.rest_filter import filter_rest


def append_data(raw_data_path, data_processed_path, pre_process_feature_path, segment_size):
    """
    Appends the new data with the previous data of the particular user and saved as a csv file inside the corresponding user folder. The new data are pre-processed by rest filter model before appending. For new user, the data is pre-processed and saved as a csv file after creating a folder for new user.
    
    Parameters:
    -----------
        raw_data_path: root path of users containing raw acc and gyro 
        data_processed_path: root path containing the processed data
        pre_process_feature_path: path contating the pre-process rest features
        segment_size: size of the window for which features are calculated
    
    Returns:
    --------
        None
    """
    
    user_id_list        = glob(raw_data_path+'/*')

    print("Pre-processing raw dataset")
    for id_path in tqdm(user_id_list):
        path_acc  = os.path.join(id_path, 'Accel.csv')
        path_gyro = os.path.join(id_path, 'Gyro.csv')
        id        = id_path.split('/')[-1]

        recent_data       = import_raw_dataset(path_acc, path_gyro)
        recent_data       = recent_data.sort_values('Time')
    
        num_segment = np.floor(recent_data.shape[0] / segment_size).astype(np.int32)
        fil_data    = pd.DataFrame(columns=['Time', 'X_data'])

        with open("logs.txt", "a") as f:
            f.write(f"{datetime.datetime.utcnow()} | Importing data==>{id} \n")
        
        for seg in tqdm(range(num_segment)):
            start_idx = seg * segment_size
            end_idx   = start_idx  + segment_size 
            seg_data  = recent_data[start_idx : end_idx]
            is_rest   = filter_rest(seg_data, segment_size, pre_process_feature_path)

            if is_rest == 1:
                continue

            else:
                fil_data = fil_data.append({'Time':seg_data.iloc[-1].Time, 'X_data':np.array(seg_data[['gyr_x', 'gyr_y', 'gyr_z', 'acc_x', 'acc_y', 'acc_z']].values)}, ignore_index=True)

            if seg % 5000 == 0:
                with open("logs.txt", "a") as f:
                    f.write(f"{datetime.datetime.utcnow()} | Segment==>{seg} \n")


        with open("logs.txt", "a") as f:        
            f.write(f"{datetime.datetime.utcnow()} | Pre-filtered data import \n")

        pre_fil, status = pre_fildata_import(data_processed_path + str(id))

        with open("logs.txt", "a") as f:
            f.write(f"{datetime.datetime.utcnow()} | Pre-filtered data import completed!!! \n")

        # if the users have previous data then concat new data with previously stored data
        if status == True:
            total_data    = pd.concat([pre_fil, fil_data], axis = 0)
        
        else:
            total_data = fil_data

        with open("logs.txt", "a") as f:
            f.write(f"{datetime.datetime.utcnow()} | Appending Done!!! \n")


        total_data = total_data.sort_values('Time')
        filename   = os.path.join(data_processed_path, str(id), "data_fil.csv")

        with open("logs.txt", "a") as f:
            f.write(f"{datetime.datetime.utcnow()} | Sorting Done!!! \n")

        print(filename)
        with open(filename, 'wb') as f:
            dump(total_data, f)

        with open("logs.txt", "a") as f:
            f.write(f"{datetime.datetime.utcnow()} | Data saved \n")
            f.write("================================================================== \n")

def prepare_training_data(legitimate_id, path):
    """
    Prepare the dataset for training the cnn model.
    
    Parameters:
    -----------
        legitimate_data: data that has been filtered by the rest filter model
        intruders_id: list containing the Android ID other than the legitimate Android ID
        path: path of file
    
    Returns:
    --------
        Tuple of numpy array, (X_train, Y_train) suitable for feeding to cnn model.
    """
    print(path)
    intruders_id      = [i.split('/')[-1].split('.')[0] for i in glob(path+'/*')]
    print(intruders_id)
    intruders_id.remove(str(legitimate_id))

    print(f"Legitimate=>{legitimate_id}")
    print(f"Intruder=>", intruders_id)

    with open(os.path.join(path, str(legitimate_id), "data_fil.csv"), 'rb') as f:
        legitimate_data         = load(f)

    legitimate_data['User'] = 1

    temp   = legitimate_data
    length = legitimate_data.shape[0]
    rand   = np.random.RandomState(seed=20)

    if len(intruders_id) > 20:
        intruders_id = rand.choice(intruders_id, 20)

    num_segment = np.floor(length / len(intruders_id)).astype(np.int32)

    for int_id in tqdm(intruders_id):
        try:
            with open(os.path.join(path, str(int_id),"data_fil.csv", 'rb')) as f:
                intruder_data         = load(f)

            intruder_data['User'] = 0
            intr_shape            = intruder_data.shape[0]
            start_x = rand.choice(range(intr_shape-num_segment - 1))
            end_x   = start_x + num_segment
            intruder_data = intruder_data[start_x : end_x]
            temp = pd.concat([temp, intruder_data], axis = 0)
        except:
            pass

    train   = shuffle(temp)
    drop    = train['X_data'].to_list()
    X       = np.stack(drop, axis = 0)
    Y       = np.array(train['User'].to_list())

    return X , Y



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass required variables')
    parser.add_argument('--id', type=str, help = 'Legitimate id')
    parser.add_argument('--raw_data_path', type = str, help='Root raw dataset path', default='../dataset/raw_data')
    parser.add_argument('--data_processed_path', type = str, help='Root data processed path', default='../dataset/data_processed')
    parser.add_argument('--pre_process_feature_path', type=str, help='Path to the pre-process rest features', default='../weights/')
    parser.add_argument('--cnn_training_datset_path', type = str, help='Root path for cnn training datset', default='../dataset/cnn_training_dataset')
    parser.add_argument('--segment_size', type = int, help = 'Number of segment per window', default = 200)
    parser.add_argument("--pre_process_raw_data", action='store_const', const=True, default=False)

    args = parser.parse_args()
    
    if args.pre_process_raw_data:
        append_data(args.raw_data_path, args.data_processed_path, args.pre_process_feature_path, args.segment_size)

    X , Y = prepare_training_data(args.id, args.data_processed_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X,  Y, test_size=0.2, random_state=42,stratify=Y)
    save_path = os.path.join(args.cnn_training_datset_path,str(args.id))
    save_data(X_train, save_path+'/X_train.csv')
    save_data(X_test, save_path+'/X_test.csv')
    save_data(Y_train, save_path+'/Y_train.csv')
    save_data(Y_test, save_path+'/Y_test.csv')

    print("Dataset saved")
    
