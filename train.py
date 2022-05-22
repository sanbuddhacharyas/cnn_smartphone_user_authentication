import os
from models.cnn_model import cnn
import argparse
from utils.load_dataset import load_cnn_training_dataset
from utils.compute_metrics import find_optimal_threhold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass required variables')
    parser.add_argument('--id', type=str, help = 'Legitimate id')
    parser.add_argument('--kernel_size', type=int, help='Kernel size', default=5)
    parser.add_argument('--window_size', type=str, help = 'Number of samples per window', default=200)
    parser.add_argument('--lr', type=str, help = 'Learning rate', default=0.001)
    parser.add_argument('--num_filters', type=str, help = 'number of filters to use', default=64)
    parser.add_argument('--epoch', type = int, help="Number of epoch to run", default=500)
    parser.add_argument('--dataset_path', type = str, help="Root path of your dataset", default='./dataset/cnn_training_dataset')

    args = parser.parse_args()

    print(f"Legitimate_id=>{args.id} | Kernel_size=>{args.kernel_size} | Window_size=>{args.window_size}")

    # Load dataset
    X_train     = load_cnn_training_dataset(os.path.join(args.dataset_path, args.id, 'X_train.csv'))
    Y_train     = load_cnn_training_dataset(os.path.join(args.dataset_path, args.id, 'Y_train.csv'))
    X_test      = load_cnn_training_dataset(os.path.join(args.dataset_path, args.id, 'X_test.csv'))
    Y_test      = load_cnn_training_dataset(os.path.join(args.dataset_path, args.id, 'Y_test.csv'))

    # Create CNN model
    cnn_model        = cnn(num_filters = args.num_filters, kernel_size = args.kernel_size, num_sample=args.window_size, learning_rate = args.lr)
    
    # Train CNN classifier model
    classifer        = cnn_model.train(X_train, Y_train, X_test, Y_test, args.id, epochs=args.epoch)
    
    # Evaluate accuracy on testing data
    pred             = classifer.predict(X_test)

    find_optimal_threhold(Y_test, pred)
