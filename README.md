This repository includes the code for the published paper [```CNN-Based Continuous Authentication of Smartphones Using Mobile Sensors.```](https://www.ijirae.com/volumes/Vol9/iss-08/37.AUAE10083.pdf) It features the model architecture, data collection tools, and methods for feature extraction. <br>
[Paper Link](https://www.ijirae.com/volumes/Vol9/iss-08/37.AUAE10083.pdf)

### Installation instructions
1. Create a new conda environment
    ```conda create -n userauth python=3.8```
2. Activate the environment
    ```conda activate userauth```
3. Install the required dependencies
    ```pip install -r requirements.txt```

### Prepare training dataset for REST filter model
```cd prepare_dataset```

You can directly download our dataset [rest_data](https://drive.google.com/file/d/1L0k4S51m0QreSPeh8UvDZjrO-79uN5cH/view?usp=sharing)
or Create your dataset with our [app](https://drive.google.com/file/d/14MLhiIHaFmJIiJlIwK3JtkvA60hOCj4G/view?usp=sharing).  
![image](https://drive.google.com/uc?export=view&id=1dlXuthgsF9G6XTcSUdqyt9HKLyf90jR6)

*Note: accel.csv and gyro.csv is store at Android/data/com.example.user_auth/files*   
*Rename accel.csv -> Accel.csv and gyro.csv->Gyro.csv.*   
a. Create dataset with app:  
    Collect REST data:  
    &emsp; Place your phone in table and click *Start* to collect the data untill it reaches 5MB and change the position  
        (1) Upright, 2) Downface, 3) Upface) make REST folder maintain the metioned directory structure  
    \
    Collect MOTION data:  
    &emsp; Start the app and perform following activities  
    &emsp;1.  scrolling  
    &emsp;2.  talking in phone  
    &emsp;3.  chatting  
    &emsp;4.  Walking  
    &emsp;5.  Jumping
    etc.


    Directory structure:
    dataset
        └── rest_data
            ├── raw
            │   ├── test
            │   │   ├── MOTION
            │   │   │   ├── 1
            │   │   │   │   ├── Accel.csv
            │   │   │   │   └── Gyro.csv
            │   │   │   ├── 2
            │   │   │   │   ├── Accel.csv
            │   │   │   │   └── Gyro.csv
            │   │   │   └── 3
            │   │   │       ├── Accel.csv
            │   │   │       └── Gyro.csv
            │   │   └── REST
            │   │       ├── 1
            │   │       │   ├── Accel.csv
            │   │       │   └── Gyro.csv
            │   │       ├── 2
            │   │       │   ├── Accel.csv
            │   │       │   └── Gyro.csv
            │   │       └── 3
            │   │           ├── Accel.csv
            │   │           └── Gyro.csv
            │   └── train
            │       ├── MOTION
            │       │   ├── 1
            │       │   │   ├── Accel.csv
            │       │   │   └── Gyro.csv
            │       │   ├── 2
            │       │   │   ├── Accel.csv
            │       │   │   └── Gyro.csv
            │       │   ├── 3
            │       │   │   ├── Accel.csv
            │       │   │   └── Gyro.csv
            │       │   └── 4
            │       │       ├── Accel.csv
            │       │       └── Gyro.csv
            │       └── REST
            │           ├── 1
            │           │   ├── Accel.csv
            │           │   └── Gyro.csv
            │           ├── 2
            │           │   ├── Accel.csv
            │           │   └── Gyro.csv
            │           ├── 3
            │           │   ├── Accel.csv
            │           │   └── Gyro.csv
            │           └── 4
            │               ├── Accel.csv
            │               └── Gyro.csv



Run following code to prepare rest filter training dataset:  
This code creates *dataset/rest_data/train* folder.
```bash
python  prepare_rest_filter_dataset.py
```
### Train REST filter model
You can directly download the models: [Rest](https://drive.google.com/file/d/1_CLDVKVbcZkcqCSNuQfCmcQgQlfw8PKY/view?usp=sharing)

```bash
Model directory structure
weights
└── Rest
    ├── Features
    │   ├── coorel_features.txt
    │   ├── selector.joblib
    │   └── std_scaler.joblib
    └── Model
        └── model.sav


```

To train the model, run the following command.
```bash
cd ..
python train_restfilter.py
```
This command creates model.sav at *weights/Rest/Model/* and pre-processing rest features  
*coorel_features.txt, selector.joblib and  std_scaler.joblib* at location *weights/Rest/Features*

The *model.sav , coorel_features.txt, selector.joblib and  std_scaler.joblib* are used to filter the rest or motion data during CNN data perparation and testing. 


### Prepare training dataset for CNN model:
You can directly download our dataset [raw_data](https://drive.google.com/file/d/1_mqModB8Q5n_EKbUrjJgrgqp8iKKI0Y6/view?usp=sharing), [data_processed](https://drive.google.com/file/d/1TGAU-u8Sko5nZIeRKSvoY88OUiJuE2D8/view?usp=sharing) and [cnn_training_dataset](https://drive.google.com/file/d/1WahW47_2_7v2EuwOUZGt61J3W2Lu9gnl/view?usp=sharing) for testing and can assign as intruder 


Create dataset with app:  
&emsp; Start the app, it will automatically collects the data when you are active.  
&emsp; copy the Accel.csv and Gyro.csv in the following directory structure

    Directory structure:
    dataset
    └── raw_data
        ├── 37905f086c46d416
        │   ├── Accel.csv
        │   └── Gyro.csv
        ├── 79435f784571ade9
        │   ├── Accel.csv
        │   └── Gyro.csv
        ├── 9b09051ccc1f3dc4
        │   ├── Accel.csv
        │   └── Gyro.csv
        └── f66e251ee7c7dbae
            ├── Accel.csv
            └── Gyro.csv

Run the following code to prepare CNN training dataset
```bash
cd prepare_dataset
python prepare_cnn_dataset.py --id 'your_id_name' --pre_process_raw_data
```
It takes 1 - 2 hour according to the size of dataset.  

This code creates data_processed and cnn_training_dataset folder at *dataset*  
where data_processed contains pre-processed data for each users and  the cnn_training_dataset  
contains the training dataset for the user (--id 'your_id_name') and all the users are intruder for  
'your_id_name'.

Once the raw data is pre-processed for all the available users. You don't need to pass --pre_process_raw_data.  
you can directly use following code
```bash
cd prepare_dataset
python prepare_cnn_dataset.py --id 'your_id_name'
```  

2. For CNN model

### Train the model
To train the model, run the following command.
```bash
    python train.py --id user_id --kernel_size kernel_size --window_size window_size --lr learning_rate --num_filters number_of_filters_to_use --epoch number_of_epochs_to_train --dataset_path root_path_of_the_dataset
```
Example:
```bash
    python train.py --id 37905f086c46d416 --epoch 50
```

### Test your model
You can prepare testing dataset with the above method
To run the testing code:
```bash
python test.py --id 'your_id' --X 'dataset path' --Y 'dataset label path' --model_path 'Trained model checkpoint path (.h5)'
```
Example:
```bash
python test.py --id 37905f086c46d416 --X ./dataset/cnn_training_dataset/37905f086c46d416/X_test.csv --Y ./dataset/cnn_training_dataset/37905f086c46d416/Y_test.csv --model_path ./weights/37905f086c46d416/Models/cp.h5
```
