from joblib import load, dump

def load_dataset(input_file):
    with open(input_file, 'rb') as f:      #Import rest data
        return load(f)

    