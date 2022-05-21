import pickle
from joblib import dump, load
import tsfel
import numpy as np

def fill_missing_values(df):
    """ Handle missing data. Replace inf, -inf and NaN values with mean of the corresponding columns.
    Parameters:
    -----------
        df: pandas dataframe containing the features
    Returns:
    --------
	    Pandas dataframe with -inf and inf values replaced by the mean of the columns
    """

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

def pre_processing_rest_features(X_test, path):
    """
    Fills missing values in the dataframe, remove correlated features and low variance features as well as normalize the features using standard score, 
    z = (x - u) / s
    Parameters:
    -----------
        X_test: pandas dataframe
        path: path contating the pre-process rest features
    Returns:
    --------
        Numpy array after preprocessing on the dataframe.
    """

    # Handling eventual missing values from the feature extraction
    X_test = fill_missing_values(X_test)

    # Highly correlated features are removed
    with open(path+"/Rest/Features/coorel_features.txt", 'r') as f:
        corr_features= f.read()

    X_test.drop(eval(corr_features), axis=1, inplace=True)
   
    # Remove low variance features
    # with open(, "rb") as f:
    selector = load(path+"/Rest/Features/selector.joblib")

    X_test  = selector.transform(X_test)
    
    # Normalising Features
    # with open(path+, "rb") as f:
    std_scaler = load(path+"Rest/Features/std_scaler.joblib")
        
    nX_test   = std_scaler.transform(X_test)

    return nX_test

def extract_rest_features(train_drop, segment_size):
    """
    Extracts necessary features from raw data in the dataframe. Time column in the dataframe is dropped and features are calculated at 50 Hz sampling frequency and window size of segment_size.
    
    Parameters:
    -----------
        train_drop: Pandas dataframe
        segment_size: size of the window for which features are calculated
    
    Returns:
    --------
        Pandas dataframe containing the features extracted.
    """

    # dataset sampling frequency
    fs = 50
    cfg_file   = tsfel.get_features_by_domain()
    train_drop = train_drop.drop('Time', axis = 1)
    # Get features
    features = ['Absolute energy', 'Area under the curve', 'Interquartile range', 'Kurtosis', 'Max', 'Mean', 'Min', 'Median absolute deviation', 'Standard deviation',  # 'Fundamental frequency',
                'Maximum frequency', 'Entropy', 'Negative turning points', 'ECDF Percentile', 'Median absolute diff', 'Spectral distance', 'Wavelet energy', 'Wavelet variance', 'Power bandwidth']
    require_features = {'spectral': {}, 'statistical': {}, 'temporal': {}}
    for domain in require_features.keys():
        for fea in features:
            if fea in cfg_file[domain].keys():
                r = cfg_file[domain][fea]
                require_features[domain][fea] = r

    train_drop = tsfel.time_series_features_extractor(require_features, train_drop, fs=fs, window_size = segment_size, verbose=0)

    return train_drop

def filter_rest(input, segment_size, pre_process_feature_path):
    """
    Predicts whether data segment is of rest or motion by rest filter model stored in S3.
    Parameters:
    -----------
        input: pandas dataframe containing raw accelerometer and gyroscope data
        segment_size: size of the window for which features are calculated
        pre_process_feature_path: path contating the pre-process rest features

    Returns:
    --------
        Boolean (if rest True \  if motion False).
    
    """
    with open(pre_process_feature_path+'Rest/Model/model.sav', 'rb') as f:
        classifier = pickle.load(f)

    rest_features     = extract_rest_features(input, segment_size)
    X_test            = pre_processing_rest_features(rest_features, pre_process_feature_path)
    out               = classifier.predict(X_test)

    return out