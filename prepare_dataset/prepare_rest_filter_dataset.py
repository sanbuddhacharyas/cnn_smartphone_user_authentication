from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from joblib import load, dump
import tsfel
import sys
import pandas as pd
from glob import glob
import numpy as np
from sklearn.utils import shuffle
import os
import argparse

sys.path.insert(0, '../')
from utils.rest_filter import fill_missing_values 
from utils.load_dataset import save_data

def extract_features_train(train, path_dir, window_size):
    # dataset sampling frequency
    f_s = 50

    # Extract excel info
    cfg_file = tsfel.get_features_by_domain()
    try:
        train.drop(['Time'], axis=1, inplace=True)

    except:
        pass

    train_drop = train.drop('act', axis=1)
    print(train_drop.columns)
    # Get features
    features = ['Absolute energy', 'Area under the curve', 'Interquartile range', 'Kurtosis', 'Max', 'Mean', 'Min', 'Median absolute deviation',  'Standard deviation', 'Maximum frequency', 'Entropy',  'Negative turning points',  'ECDF Percentile', 'Median absolute diff', 'Spectral distance', 'Wavelet energy', 'Wavelet variance', 'Power bandwidth']
    require_features = {'spectral' :{}, 'statistical' :{}, 'temporal':{}}
    for domain in require_features.keys():
        for fea in features:
            if fea in cfg_file[domain].keys():
                r = cfg_file[domain][fea]
                require_features[domain][fea] = r

    #Extract hand picked features
    X_train = tsfel.time_series_features_extractor(require_features, train_drop, fs = f_s, window_size = window_size )
    num_data     = np.floor(train.shape[0] / window_size ).astype('int32')
    Y_train = [train.act[num*window_size :num*window_size +window_size ].mode().values[0] for num in range(num_data)]

    X_train          = pre_processing(X_train, path_dir)
    X_train, Y_train = shuffle(X_train, Y_train, random_state = 42)

    return X_train, Y_train

def extract_features_test(test, path_dir, window_size):
    # dataset sampling frequency
    f_s = 50

    # Extract excel info
    cfg_file = tsfel.get_features_by_domain()
    try:
        test.drop(['Time'], axis=1, inplace=True)

    except:
        pass

    test_drop = test.drop('act', axis=1)
    print(test_drop.columns)
    # Get features
    features = ['Absolute energy', 'Area under the curve', 'Interquartile range', 'Kurtosis', 'Max', 'Mean', 'Min', 'Median absolute deviation',  'Standard deviation', 'Maximum frequency', 'Entropy',  'Negative turning points',  'ECDF Percentile', 'Median absolute diff', 'Spectral distance', 'Wavelet energy', 'Wavelet variance', 'Power bandwidth']
    require_features = {'spectral' :{}, 'statistical' :{}, 'temporal':{}}
    for domain in require_features.keys():
        for fea in features:
            if fea in cfg_file[domain].keys():
                r = cfg_file[domain][fea]
                require_features[domain][fea] = r

    #Extract hand picked features
    X_test = tsfel.time_series_features_extractor(require_features, test_drop, fs = f_s, window_size = window_size )
    X_test         = pre_processing_rest_features(X_test, path_dir)

    num_data     = np.floor(test.shape[0] / window_size ).astype('int32')
    Y_test = [test.act[num*window_size :num*window_size +window_size ].mode().values[0] for num in range(num_data)]
    
    return X_test, Y_test
    
def pre_processing_rest_features(X_test, path):
    # Handling eventual missing values from the feature extraction
    X_test = fill_missing_values(X_test)

    # Highly correlated features are removed

    with open(path+"/Rest/Features/coorel_features.txt", 'r') as f:
        corr_features= f.read()

    X_test.drop(eval(corr_features), axis=1, inplace=True)
   
    # Remove low variance features
    with open(path+"/Rest/Features/selector.joblib", "rb") as f:
        selector = load(f)
    X_test  = selector.transform(X_test)

    # Normalising Features
    with open(path+"/Rest/Features/std_scaler.joblib", "rb") as f:
        std_scaler = load(f)
    nX_test   = std_scaler.transform(X_test)

    return nX_test

def pre_processing(X_train, path):
    # Handling eventual missing values from the feature extraction
    X_train = fill_missing_values(X_train)

    # Highly correlated features are removed
    corr_features = tsfel.correlated_features(X_train)
    X_train.drop(corr_features, axis=1, inplace=True)
    
    os.makedirs(path+"/Rest/Features/", exist_ok=True)
    with open(path+"/Rest/Features/coorel_features.txt", 'w') as f:
        f.write(str(corr_features))


    # Remove low variance features
    selector = VarianceThreshold()
    X_train  = selector.fit_transform(X_train)
    with open(path+"/Rest/Features/selector.joblib", "wb") as f:
        dump(selector, f)

    # Normalising Features
    std_scaler = preprocessing.StandardScaler()
    nX_train = std_scaler.fit_transform(X_train)
    with open(path+"/Rest/Features/std_scaler.joblib", "wb") as f:
        dump(std_scaler, f)
   
    return nX_train

def return_merge(train_dataset, max_datasize):
    all_merge_data = []
    for train_path in train_dataset:
        acc = pd.read_csv(train_path+'/Accel.csv', names = ['Time', 'acc_x','acc_y', 'acc_z'])
        gyr = pd.read_csv(train_path+'/Gyro.csv' , names = ['Time', 'gyr_x','gyr_y', 'gyr_z'])

        acc_gyro = pd.merge(acc, gyr, on='Time')
        print(train_path.split('/')[-2])
        acc_gyro['act'] = train_path.split('/')[-2]

        all_merge_data.append(acc_gyro[:max_datasize])

    all_merge_data = pd.concat(all_merge_data)
    return all_merge_data[['Time','gyr_x', 'gyr_y', 'gyr_z','acc_x', 'acc_y' ,'acc_z', 'act']]

def combine_raw_dataset(path):
    train_dataset = glob(path + '/train/*/*')
    test_dataset  = glob(path + '/test/*/*')

    print(train_dataset, test_dataset)

    max_datasize   = 40000
    train = return_merge(train_dataset, max_datasize)
    test  = return_merge(test_dataset, max_datasize)


    return train, test

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Pass required variables')
    parser.add_argument('--raw_data_path', type = str, help='Root raw dataset path', default='../dataset/rest_data/raw')
    parser.add_argument('--rest_train_path', type = str, help='Root path for cnn training datset', default='../dataset/rest_data/train')
    parser.add_argument('--pre_process_feature_path', type=str, help='Path to the pre-process rest features', default='../weights/')
    parser.add_argument('--segment_size', type = int, help = 'Number of segment per window', default = 200)

    args = parser.parse_args()


    train, test = combine_raw_dataset(args.raw_data_path)
    X_train, Y_train = extract_features_train(train, args.pre_process_feature_path, args.segment_size)
    X_test, Y_test   = extract_features_test(test, args.pre_process_feature_path, args.segment_size)

    # Save the training dataset for rest filter
    save_data(X_train, args.rest_train_path+'/X_train.csv')
    save_data(X_test, args.rest_train_path+'/X_test.csv')
    save_data(Y_train, args.rest_train_path+'/Y_train.csv')
    save_data(Y_test, args.rest_train_path+'/Y_test.csv')



