import argparse
from utils.load_dataset import load_cnn_training_dataset
from tensorflow.keras.models import load_model
from utils.compute_metrics import find_optimal_threhold




import sys
sys.path.insert(0, '../')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass required variables')
    parser.add_argument('--id', type=str, help = 'Legitimate id')
    parser.add_argument('--X', type = str, help="Data path")
    parser.add_argument('--Y',  type = str, help="label path")
    parser.add_argument('--model_path',  type = str, help="label path")
    args = parser.parse_args()

    X     = load_cnn_training_dataset(args.X)
    Y     = load_cnn_training_dataset(args.Y)

    classifier       = load_model(args.model_path)
    pred             = classifier.predict(X)


    find_optimal_threhold(Y, pred)

    target_names=['Intruder', 'Legitimate']
    