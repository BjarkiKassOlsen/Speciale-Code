
from project_imports import *

# from importlib import reload
import generate_graphs
import functions
import Download_and_process_data
import train
import custom_model
import custom_dataset

import psutil

def print_memory_stats(epoch):
    """Prints GPU and CPU memory usage."""
    # GPU Memory Stats
    if torch.cuda.is_available():
        print(f"\n[Epoch {epoch}] GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"[Epoch {epoch}] GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        print(f"[Epoch {epoch}] GPU Max Memory Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")

    # CPU Memory Stats
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"[Epoch {epoch}] CPU Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")  # Resident Set Size (RSS) in MB

run = neptune.init_run(
    project="bjarki/Speciale",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzc0ZjBjMy0wYzBjLTQwMGYtODExYS1iNDM1MjAxZDdlNWMifQ==",
)  # your credentials

###################################
##  Run CNN on US Market stocks  ##
###################################

model_run_nr = os.getenv("SLURM_ARRAY_TASK_ID")

# Specify the data path
table = 'I20VolTInd20'
market = 'US'
ws = 20

# Setup the path to the dataset
hdf5_dataset_path = f'{DATA_PATH}/{market}/{table}_dataset.h5'

# Retrieved from previous local run
computed_mean = 0.1057652160525322
computed_std = 0.3043188750743866

####

print(f"Mean: {computed_mean}, Std: {computed_std}")


# Define transformations
custom_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize(mean=[computed_mean], std=[computed_std]),
])





def rolling_window_training(hdf5_dataset_path, model_class, transform, all_dates, num_in_batch=128, num_workers=8, n_epochs=5, lr=1e-4, train_ratio=0.8, run = None):
    """Performs rolling window training and testing on the provided HDF5 dataset."""
    all_epoch_stats = []  # Store training stats for each window
    # Initialize list to collect DataFrames
    results = []
    results_train = [] # List to store the predictions made, for the training period, to be used for our combined model.
    
    train_start = 199301  # Initial training start year as an integer
    train_window_years = 7
    test_window = 1
    
    # Initialize the model (use the actual class reference)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_class == 'CNN':
        # Instantiate the CNN model
        model = custom_model.CNNModel(
            win_size=20,
            inplanes=64,
            drop_prob=0.50,
            batch_norm=True,
            xavier=True,
            lrelu=True,
            bn_loc="bn_bf_relu",
            regression_label=False
        )
    
    # Send model to the GPU
    model.to(device)

    # Initialize the dataset
    dataset_train = custom_dataset.GraphDataset(path=hdf5_dataset_path, transform=transform, mode='train', model=model_class)
    dataset_test = custom_dataset.GraphDataset(path=hdf5_dataset_path, transform=transform, mode='test', model=model_class)
    
    while  train_start + train_window_years * 100 < 202201:
        # Define train and test period dates in integer format
        train_end = train_start + train_window_years * 100
        test_end = train_end + test_window     

        print(f"Training from {train_start} to {train_end}, Predicting from {train_end} to {test_end}")

        # Filter the dataset using the preloaded dates for training
        indices_train = [i for i, date in enumerate(all_dates) if train_start <= int(date.decode('utf-8')) < train_end]
        
        # Filtered dataset based on date range for training
        graph_dataset_train = torch.utils.data.Subset(dataset_train, indices_train)
        
        # Split into train and validation sets
        train_loader_size = int(len(graph_dataset_train) * train_ratio)
        valid_loader_size = len(graph_dataset_train) - train_loader_size

        # Split the dataset using the precomputed indices
        train_loader, valid_loader = random_split(graph_dataset_train, [train_loader_size, valid_loader_size])

        # DataLoader for train and validation
        train_loader = DataLoader(dataset=train_loader, batch_size=num_in_batch, shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_loader, batch_size=num_in_batch, shuffle=True, num_workers=num_workers, pin_memory=True)
        
        # Filter the dataset using the preloaded dates for testing
        indices_test = [i for i, date in enumerate(all_dates) if train_end <= int(date.decode('utf-8')) < test_end]
        
        # Filtered dataset based on date range for testing
        graph_dataset_test = torch.utils.data.Subset(dataset_test, indices_test)
        
        # DataLoader for testing
        test_loader = DataLoader(graph_dataset_test, batch_size=num_in_batch, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Set model to train
        model.train()
        
        # Train the model
        epoch_stats, best_validate_metrics, model = train.train_n_epochs(
            n_epochs=n_epochs,
            model=model,
            pred_win=20,
            train_loader=train_loader,
            valid_loader=valid_loader,
            early_stop=True,
            early_stop_patience=5,
            lr=lr,
            regression_label=False,
            run=run
        )

        # Store the epoch statistics
        all_epoch_stats.append(epoch_stats)

        # Print memory usage after training
        print_memory_stats(epoch="After Training")
        # Memory cleanup before predicting
        torch.cuda.empty_cache()
        gc.collect()

        # Set model to evalutaion for prediction
        model.eval()
        
        # Disable gradient computation for inference
        with torch.no_grad():
            # Wrap your data loader with tqdm for a progress bar
            for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Predicting'):
                images = batch['image'].to(device)  # Ensure images are on the same device as the model
                outputs = model(images)  # Get logits from the model

                # Record logits along with permno and date for each item in the batch
                batch_results = pd.DataFrame({
                    'permno': batch['permno'],
                    'date': batch['yyyymm'],
                    'label': batch['label'].cpu().numpy(),
                    'ME': batch['ME'].cpu().numpy(),
                    'neg_ret': outputs[:, 0].cpu().numpy(),
                    'pos_ret': outputs[:, 1].cpu().numpy()
                })

                results.append(batch_results)
                
                # Explicitly delete the batch_results and run garbage collection
                del batch_results
                torch.cuda.empty_cache()
                gc.collect()
                
        # Move the window one month forward
        if train_start % 100 != 12:
            train_start += test_window
        else:
            train_start += 100 - 11
            break
            
        # Move the training window forward by 5 years for the next iteration
        # train_start += test_window_years  # Move start forward by 5 years

        # # Check if we are beyond the dataset's final period (e.g., after 2022)
        # if train_end >= 2022:
        #     break

    # # Only run this block, when all the training is done
    # # DataLoader for train and validation
    # dataset_train_pred = custom_dataset.GraphDataset(path=hdf5_dataset_path, transform=transform, mode='test', model=model_class)
    
    # # Filtered dataset based on date range for training
    # graph_dataset_train_pred = torch.utils.data.Subset(dataset_train_pred, indices_train)
    # train_loader = DataLoader(dataset=graph_dataset_train_pred, batch_size=num_in_batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    
    # # Disable gradient computation for inference
    # with torch.no_grad():
    #     # Wrap your data loader with tqdm for a progress bar
    #     for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Predicting'):
    #         images = batch['image'].to(device)  # Ensure images are on the same device as the model
    #         outputs = model(images)  # Get logits from the model

    #         # Record logits along with permno and date for each item in the batch
    #         batch_results = pd.DataFrame({
    #             'permno': batch['permno'],
    #             'date': batch['yyyymm'],
    #             'label': batch['label'].cpu().numpy(),
    #             'ME': batch['ME'].cpu().numpy(),
    #             'neg_ret': outputs[:, 0].cpu().numpy(),
    #             'pos_ret': outputs[:, 1].cpu().numpy()
    #         })

    #         results_train.append(batch_results)

    #         # Explicitly delete the batch_results and run garbage collection
    #         del batch_results
    #         torch.cuda.empty_cache()
    #         gc.collect()
    
    # # Collect all the in-sample results
    # results_train = pd.concat(results_train, ignore_index=True)

    # Collect all of the OOS results
    results = pd.concat(results, ignore_index=True)
    
    results.to_csv(f'{DATA_PATH}/returns/CNN_OOS_model_{model_run_nr}.csv')
    # results_train.to_csv(f'{DATA_PATH}/returns/CNN_IS_model_{model_run_nr}.csv')

    print("Rolling window training complete!")
    return all_epoch_stats, results, model

# Load the HDF5 file and extract all dates for filtering
with h5py.File(hdf5_dataset_path, 'r') as file:
    all_dates = file["dates"][:]  # Preload all dates for quick access

# Run the rolling window setup with your model class and parameters
all_stats_CNN, results_CNN, model_CNN = rolling_window_training(
    hdf5_dataset_path=hdf5_dataset_path,
    model_class='CNN',  # Specify your model class name or reference here
    transform=custom_transforms,
    all_dates = all_dates,
    num_in_batch=128,
    num_workers=8,
    n_epochs=2,
    lr=1e-4,
    train_ratio=0.8,
    run=run
)



# Initialize cumulative dictionaries for train and validation metrics
merged_stats = {'train': [], 'valid': []}

# Track the cumulative epoch number
cumulative_epoch = 0

# Iterate through each run's epoch stats and merge them
for run_stats in all_stats_CNN:
    # Update epoch number to continue from the last cumulative epoch
    for stat in run_stats['train']:
        stat['epoch'] += cumulative_epoch  # Increment the epoch numbers
        merged_stats['train'].append(stat)

    for stat in run_stats['valid']:
        stat['epoch'] += cumulative_epoch  # Increment the epoch numbers
        merged_stats['valid'].append(stat)

    # Update the cumulative epoch count
    cumulative_epoch = merged_stats['train'][-1]['epoch'] + 1  # Increment to the next epoch number



train.plot_epoch_stats(merged_stats, run=run)


# Save the model
# torch.save(model_CNN.state_dict(), f'{PROJECT_PATH}/model/LUMI_US_Market_930101_200101.pth')

# Stop the monitoring on Neptune
run.stop()
