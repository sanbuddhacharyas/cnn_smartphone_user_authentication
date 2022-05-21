### Installation instructions
1. Create a new conda environment
    ```conda create -n userauth python=3.8```
2. Activate the environment
    ```conda activate userauth```
3. Install the required dependencies
    ```pip install -r requirements.txt```

### Prepare data 
```cd prepare_dataset```
1. For rest filter model

2. For CNN model

### Train the model
To train the model, run the following command.
    ```python train.py --id user_id --kernel_size kernel_size --window_size window_size --lr learning_rate --num_filters number_of_filters_to_use --epoch number_of_epochs_to_train --dataset_path root_path_of_the_dataset```