import tensorflow.keras as keras   
from  tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Input, BatchNormalization, ReLU, Dense, Dropout
import tensorflow.compat.v1 as tf
import os

class CustomCallback(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        local_path        =  os.path.dirname(os.getcwd()) + '/'
        keys = list(logs.keys())
        out  = "Starting training; got log keys: {}".format(keys)
        print(out)
        with open(local_path + "logs.txt", "a") as f:
            f.write(out)

    def on_epoch_end(self, epoch, logs=None):
        local_path        =  os.path.dirname(os.getcwd()) + '/'
        out =  "The average loss for epoch {} is {:7.2f}.".format(
                epoch, logs["loss"])
        print(out)
        with open(local_path + "logs.txt", "a") as f:
            f.write(out) 
        

class cnn:
    def __init__(self, num_filters = 64, kernel_size = 7, num_sample=200, learning_rate = 0.001):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_sample  = num_sample
        self.input_shape = (num_sample, 6)
        self.learning_rate    = learning_rate
        self.id          = id

    def create_model(self):

        print(self.kernel_size)
        model = keras.Sequential()
        model.add(Input(shape = self.input_shape))
        model.add(Conv1D(self.num_filters, self.kernel_size,  activation=None))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Conv1D(int(self.num_filters / 2) , self.kernel_size, activation=None, dilation_rate=2))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Conv1D(int(self.num_filters / 4), self.kernel_size,  activation=None, dilation_rate=3))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.2))
        
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Dense(16))
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Dense(1, activation ='sigmoid'))
    
        return model

    def train(self, train_x, train_y, X_test_data, Y_test_data, id, path, fs, resume_checkpoint = False, epochs = 500):
        
        path     = os.getcwd() + '/'
        filename = str(id)+'/Models/'
        checkpoint_path = path + filename + "cp.ckpt"

        self.model = self.create_model()
 
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path , save_weights_only=True, verbose=1)

        self.model.compile(loss = 'binary_crossentropy', optimizer= tf.train.AdamOptimizer(learning_rate = self.learning_rate),  metrics=['accuracy'])
        
        if resume_checkpoint:
            try:
                self.model.load_weights(checkpoint_path)

            except:
                print("Checkpoint Not available")

        self.model.fit(train_x, train_y,  epochs = epochs, validation_data=(X_test_data, Y_test_data), batch_size = 32, verbose = 1, callbacks = [cp_callback])
        
        return self.model