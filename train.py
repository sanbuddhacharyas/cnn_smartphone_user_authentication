import os
import tensorflow.keras as keras 
import tensorflow.compat.v1 as tf
from models.cnn_model import cnn
import argparse
import numpy as np
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass required variables')
    parser.add_argument('--id', type=str, help = 'Legitimate id')
    parser.add_argument('--kernel_size', type=int, help='Kernel size', default=5)
    parser.add_argument('--window_size', type=str, help = 'Number of samples per window', default=200)
    parser.add_argument('--lr', type=str, help = 'Learning rate', default=0.001)
    parser.add_argument('--num_filters', type=str, help = 'number of filters to use', default=64)

    args = parser.parse_args()

    print(f"Legitimate_id=>{args.legitimate_id} | Kernel_size=>{args.kernel_size} | Window_size=>{args.window_size}")

    # Load dataset
    with fs.open(path+id+'X_test.csv', 'rb') as f:
        X_test = load(f)

    with fs.open(path+id+'Y_test.csv', 'rb') as f:
        Y_test = load(f)
        
    cnn_model        = cnn(num_filters = args.num_filters, kernel_size = args.kernel_size, num_sample=args.window_size, learning_rate = args.lr)
    classifer        = cnn_model.train(X_train_data, Y_train_data, X_test_data, Y_test_data, legitimate, path, fs, epochs=500)
    pred             = classifer.predict(X_test_data)
    pred             = pred>0.8
    accuracy         = accuracy_score(Y_test_data.astype(np.int8), pred.astype(np.int8))

    print(f"Accuracy=>{accuracy}")
