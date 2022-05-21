import os
import tensorflow.keras as keras 
import tensorflow.compat.v1 as tf
from models.cnn_model import cnn
import argparse

parser = argparse.ArgumentParser(description='Pass required variables')
parser.add_argument('--kernel_size', type=int, help='Kernel size')
parser.add_argument('--id', type=str, help = 'Legitimate id')

args = parser.parse_args()


