# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:43:38 2024

@author: bjark
"""


import sys
sys.path.append('C:/Users/bjark/Documents/AU/Kandidat/4. Semester/Code/Speciale-Code')

from project_imports import *

from importlib import reload
import generate_graphs
import functions
import Download_and_process_data

import train
reload(train)

# Fetch all the data and process it appropriately, finally save to hdf5 file
# Download_and_process_data.update()

# ============ START TRAINING ==============


# Specify the data path
table = 'I20VolTInd20'
market = 'US'
ws = 20

# Setup the path to the dataset
hdf5_dataset_path = f'{DATA_PATH}/{market}/{table}_dataset.h5'


# See example of chart
generate_graphs.show_single_graph(hdf5_dataset_path, 10000)
generate_graphs.show_single_graph(hdf5_dataset_path, 150001)


import custom_model
import custom_dataset

# Reload the modules
reload(custom_model)
reload(custom_dataset)

# Initialize the model (use the actual class reference)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# scp "C:\Users\bjark\Documents\AU\Kandidat\4. Semester\Code\Speciale-Code\data\US\I20VolTInd20_train.h5" olsenbja@lumi.csc.fi:/scratch/project_465001092/Speciale-Code/data/US
# Load the dataset without transformations for now
# graph_dataset_init = custom_dataset.GraphDataset(path=hdf5_train_path, transform=transforms.ToTensor(), mode='train')

# init_loader = DataLoader(graph_dataset_init, batch_size=128, shuffle=False, num_workers=4)

# # Initialize variables for calculating mean and std
# mean = 0.0
# std = 0.0
# n_samples = 0

# for batch in tqdm(init_loader, desc="Calculating mean and std"):
#     images = batch['image']
#     batch_samples = images.size(0)  # Batch size (number of images in the batch)
#     images = images.view(batch_samples, images.size(1), -1)  # Flatten the images to (batch_size, channels, height*width)
    
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)
#     n_samples += batch_samples

# mean /= n_samples
# std /= n_samples

# # Convert the computed mean and std from tensor to float
# computed_mean = mean.item()
# computed_std = std.item()

#### From previous run 

# ### More price steps
# computed_mean = 0.08589867502450943
# computed_std = 0.27876555919647217

### No added steps, but continuous MA line
computed_mean = 0.1057652160525322
computed_std = 0.3043188750743866

####

print(f"Mean: {computed_mean}, Std: {computed_std}")


# Define transformations
custom_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize(mean=[computed_mean], std=[computed_std]),
])


def rolling_window_training(hdf5_dataset_path, model_class, transform, all_dates, num_in_batch=128, num_workers=8, n_epochs=5, lr=1e-4, train_ratio=0.8):
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

        # # Convert the dates to `yyyymm` format for date filtering
        # train_start_int = train_start * 100 + 1
        # train_end_int = train_end * 100 + 1      
        # test_end_int = test_end * 100 + 1        

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
            regression_label=False
        )

        # Store the epoch statistics
        all_epoch_stats.append(epoch_stats)
        
        
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
                
        
        # Only run this block the very first iteration
        if train_start == 199301:
            
            # DataLoader for train and validation
            dataset_train_pred = custom_dataset.GraphDataset(path=hdf5_dataset_path, transform=transform, mode='test', model=model_class)
            
            # Filtered dataset based on date range for training
            graph_dataset_train_pred = torch.utils.data.Subset(dataset_train_pred, indices_train)
            train_loader = DataLoader(dataset=graph_dataset_train_pred, batch_size=num_in_batch, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            
            # Disable gradient computation for inference
            with torch.no_grad():
                # Wrap your data loader with tqdm for a progress bar
                for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Predicting'):
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

                    results_train.append(batch_results)
            
            results_train = pd.concat(results_train, ignore_index=True)
                
        # Move the window one month forward
        if train_start % 100 != 12:
            train_start += test_window
        else:
            train_start += 100 - 11
            
        # Move the training window forward by 5 years for the next iteration
        # train_start += test_window_years  # Move start forward by 5 years

        # # Check if we are beyond the dataset's final period (e.g., after 2022)
        # if train_end >= 2022:
        #     break
        
    results = pd.concat(results, ignore_index=True)
    
    results.to_csv(f'{DATA_PATH}/returns/CNN_OOS.csv')
    results_train.to_csv(f'{DATA_PATH}/returns/CNN_IS.csv')

    print("Rolling window training complete!")
    return all_epoch_stats, results

# Load the HDF5 file and extract all dates for filtering
with h5py.File(hdf5_dataset_path, 'r') as file:
    all_dates = file["dates"][:]  # Preload all dates for quick access

# Run the rolling window setup with your model class and parameters
all_stats_CNN, results_CNN = rolling_window_training(
    hdf5_dataset_path=hdf5_dataset_path,
    model_class='CNN',  # Specify your model class name or reference here
    transform=custom_transforms,
    all_dates = all_dates,
    num_in_batch=128,
    num_workers=8,
    n_epochs=2,
    lr=1e-4,
    train_ratio=0.8
)

# # Run the rolling window setup with your model class and parameters
# all_stats_XGB, results_XGB = rolling_window_training(
#     hdf5_dataset_path=hdf5_dataset_path,
#     model_class='XGB',  # Specify your model class name or reference here
#     transform=custom_transforms,
#     all_dates = all_dates,
#     num_in_batch=128,
#     num_workers=8,
#     n_epochs=2,
#     lr=1e-4,
#     train_ratio=0.8
# )






###### XGBOOST


def portfolio_sorting(df, pred_col='pred', ret_col='ret', me_col=None, n_portfolios=10):
    """
    General function to sort a DataFrame into portfolios based on predictions, with support for weighted sorting.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        pred_col (str): Column name of the predictions used for sorting (default is 'pred').
        ret_col (str): Column name of the actual returns to calculate portfolio returns (default is 'ret').
        me_col (str, optional): Column name of the market equity weights (default is None for equal weighting).
        n_portfolios (int): Number of portfolios to sort into (default is 10 for deciles).

    Returns:
        pd.DataFrame: DataFrame with sorted portfolios and associated returns.
    """
    # Ensure input DataFrame contains the necessary columns
    if pred_col not in df.columns or ret_col not in df.columns:
        raise ValueError(f"Columns '{pred_col}' and '{ret_col}' must be present in the input DataFrame.")
    
    # Create a new column for portfolio allocation based on predictions
    df = df.copy()  # Avoid modifying the original DataFrame

    # Handle weighting (optional)
    if me_col is not None:
        # Use weighted quantile binning if market equity (ME) weights are provided
        df['weight'] = df[me_col] / df[me_col].sum()
    else:
        # Use equal weights if no weights are provided
        df['weight'] = 1.0 / len(df)

    # Sort the DataFrame by predictions
    df.sort_values(by=pred_col, inplace=True)

    # Compute the cumulative sum of the weights to get weighted quantiles
    df['cum_weight'] = df['weight'].cumsum()

    # Determine the breakpoints for portfolio allocation (soft inequalities)
    # Use np.linspace to get the quantile cut points
    cut_points = np.linspace(0, 1.000001, n_portfolios + 1)

    # Assign portfolios based on cumulative weight with soft inequality handling
    df['portfolio'] = pd.cut(df['cum_weight'], bins=cut_points, labels=range(1, n_portfolios + 1), include_lowest=True, right=False)
    
    return df



# Define a function to compute the Sharpe Ratio
def compute_sharpe_ratio(data, date_col, pred_col, ret_col, me_col):
    """
    Calculate the Sharpe ratio given an array of portfolio returns.
    
    Args:
        data (array-like): Array of returns.
        
    Returns:
        float: Sharpe ratio.
    """
    
    # Apply function to each monthly group
    results_df = data.groupby(data[date_col]).apply(portfolio_sorting, pred_col=pred_col, ret_col=ret_col, me_col=me_col, n_portfolios=10)
    
    # Reset the index to make sure 'date' is treated as a column
    results_df.reset_index(drop=True, inplace=True)
    
    # Now sort by 'date' and 'quintile'
    results_df.sort_values(by=[date_col, pred_col], inplace=True)
    
    # Calculate weighted returns for each portfolio
    monthly_means = results_df.groupby([date_col, 'portfolio'], observed=True).apply(
        lambda x: np.sum(x[ret_col] * x['weight']) / x['weight'].sum()
    ).unstack()  # Unstack to get quintiles as columns

    # Calculate High-Low (H-L) monthly returns using weighted means
    monthly_means['H-L'] = monthly_means[10] - monthly_means[1]

    # Rename quintiles for clarity
    monthly_means.rename(columns={10: 'High', 1: 'Low'}, inplace=True)

    # Annualize the returns and standard deviations (weighted standard deviations)
    annualized_returns = monthly_means.mean() * 12
    annualized_stds = monthly_means.std() * np.sqrt(12)

    # Calculate the Sharpe ratios
    sharpe_ratios = annualized_returns / annualized_stds
    
    return sharpe_ratios, annualized_returns, annualized_stds

# Define the objective function for Optuna
def objective(trial, Chars_train, Labels_train_class, ret, train_start, train_end, all_dates, valid_window_years, device):
    """
    Objective function for Optuna hyperparameter tuning.
    
    Args:
        trial (optuna.Trial): A single trial object from Optuna.
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        X_valid (numpy.ndarray): Validation feature matrix.
        y_valid (numpy.ndarray): Validation labels.

    Returns:
        float: Negative Sharpe ratio.
    """
    # Suggest hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 5, 13),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01),
        'seed': 42
    }
    
    # Add GPU parameters if GPU is available
    if device == 'cuda':
        params.update({
            'device': 'cuda',  # Use GPU for prediction
        })
        
    # params = {
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'logloss',
    #     'max_depth': 20,
    #     'reg_alpha': 4,  
    #     'reg_lambda': 2,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'min_child_weight': 3,
    #     'learning_rate': 0.001,
    #     'device': 'cuda',
    #     'seed': 42
    # }
        
    results = []
    
    valid_date_hyp_run_start = train_end - valid_window_years * 100
    valid_date_hyp_run_end = valid_date_hyp_run_start + 1
    
    for i in range(1, valid_window_years*12 + 1):
        
        print('Testing validation year-month: ', valid_date_hyp_run_start)
        
        # Split into train and validation sets (last 36 months for validation)
        mask_valid_hyp = (valid_date_hyp_run_start <= np.array(all_dates, dtype=int)) & (np.array(all_dates, dtype=int) < valid_date_hyp_run_end)
        mask_train_hyp = (train_start <= np.array(all_dates, dtype=int)) & (np.array(all_dates, dtype=int) < valid_date_hyp_run_start)
        
        
        X_train = Chars_train[mask_train_hyp]
        X_valid = Chars_train[mask_valid_hyp]
        y_train = Labels_train_class[mask_train_hyp]
        y_valid = Labels_train_class[mask_valid_hyp]
        
        Labels_valid = Labels_train[mask_valid_hyp]
    
        # Create DMatrix for training and validation
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        
        # Train the model
        model = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)
        
        # Predict on validation set
        pred = model.predict(dvalid)
        
        # Combine predictions and actual returns into a DataFrame from the valid dataset, to be used for portfolio sort
        month_results = pd.DataFrame({'yyyymm': np.array(all_dates[mask_valid_hyp], dtype=int), 'ret': Labels_valid, 'pred': pred})
        
        # Append to the results list
        results.append(month_results)
        
        if i % 12 == 0:
            valid_date_hyp_run_start = (valid_date_hyp_run_start - 12) + 101
            valid_date_hyp_run_end = (valid_date_hyp_run_end - 12) + 101
        else:
            valid_date_hyp_run_start += 1
            valid_date_hyp_run_end += 1
            
    results = pd.concat(results, ignore_index=True)
    
    # Get the sharpe ratio
    sharpe_ratio, _, _ = compute_sharpe_ratio(results, date_col = 'yyyymm', pred_col='pred', ret_col='ret', me_col=None)

    # Return negative Sharpe ratio of the H-L portfolio, for minimization
    return -sharpe_ratio['H-L']




train_start = 199301
train_window_years = 7
valid_window_years = 2
test_window = 1
train_ratio = 0.8
trained_model = None

# Function to perform walk-forward hyperparameter tuning and forecasting
# def walk_forward_xgboost(hdf5_dataset_path, initial_train_start, train_window_years, test_window_years):
"""
Perform walk-forward hyperparameter tuning and forecasting using XGBoost and Optuna.

Args:
    hdf5_dataset_path (str): Path to the HDF5 dataset.
    initial_train_start (int): Initial year-month start (format: yyyymm).
    train_window_years (int): Length of the training window in years.
    test_window_years (int): Length of the test window in years.

Returns:
    dict: Dictionary of forecasts for each month in the out-of-sample period.
"""
results = []  # List to store DataFrame results
results_train = [] # List to store the predictions made, for the training period, to be used for our combined model.
all_forecasts = {}
best_param_value = 0

# Track when next tuning period is
next_tuning = train_start
tuning_years = 3

# Load all dates
with h5py.File(hdf5_dataset_path, 'r') as file:
    all_dates = file['dates'][:]

while train_start + train_window_years * 100 < 202201:
    # Define train, validation, and test periods
    train_end = train_start + train_window_years * 100
    test_end = train_end + test_window
    
        
    print(f"Training from {train_start} to {train_end}, Testing from {train_end} to {test_end}")


    mask_train = (train_start <= np.array(all_dates, dtype=int)) & (np.array(all_dates, dtype=int) < train_end)
    # Extract training and validation data
    with h5py.File(hdf5_dataset_path, 'r') as file:
        Chars_train = file['chars'][:][mask_train]
        Labels_train = file['labels'][mask_train]
    
    # Convert labels to binary classes
    Labels_train_class = np.array([1 if label > 0 else 0 for label in Labels_train])
    
    # Train the model using the best hyperparameters, re-estimate every 5 years
    if train_start == next_tuning:
    
        # Hyperparameter tuning using Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, Chars_train, Labels_train_class, Labels_train, 
                                               train_start, train_end, all_dates[mask_train], valid_window_years, device), 
                       n_trials=20, n_jobs=2)
    
        # Get best hyperparameters
        best_params = study.best_params
        
        # Check if the best params this search is better than previous
        if study.best_value < best_param_value:
            
            # Get best hyperparameters
            best_params_global = study.best_params
            
            # Update best value
            best_param_value = study.best_value
            
        # Set next tuning to 5 years forward
        next_tuning += tuning_years * 100
        
        # Restart model trained
        trained_model = None

    
    # Split into train and validation sets
    X_train_size = int(len(Labels_train_class) * train_ratio)
    X_valid_size = len(Labels_train_class) - X_train_size

    # Split the dataset using the precomputed indices
    X_train, X_valid, y_train, y_valid = train_test_split(Chars_train, Labels_train_class, test_size=X_valid_size, shuffle=True, random_state=42)

    # Create a boolean mask for the desired date range
    mask_test = (train_end <= np.array(all_dates, dtype=int)) & (np.array(all_dates, dtype=int) < test_end)
    
    # Filtered dataset based on date range for training
    with h5py.File(hdf5_dataset_path, 'r') as file:
        
        X_test = file['chars'][:][mask_test]
        y_test = file['labels'][mask_test]
        permnos = file['permnos'][mask_test]
        dates = file['dates'][mask_test]
        me = file['ME'][mask_test]
    
    # Convert labels to classes [0,1]
    # y_test = [[0,1] if label > 0 else [1,0] for label in y_test]
    y_test_class = np.array([1 if label > 0 else 0 for label in y_test])
    
    # Convert to XGBoost DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test, label=y_test_class)
    
    
    # Train the model using the best hyperparameters
    # trained_model = xgb.train(best_params_global, dtrain, evals=[(dtrain, 'train'), (dvalid, 'eval')],
    #                           early_stopping_rounds=10, num_boost_round=1000)
    
    if trained_model is None:
        # Train from scratch
        trained_model = xgb.train(best_params_global, dtrain, evals=[(dtrain, 'train'), (dvalid, 'eval')],
                                  early_stopping_rounds=10, num_boost_round=2000)
    else:
        # Update the existing model with new data
        trained_model = xgb.train(best_params_global, dtrain, evals=[(dtrain, 'train'), (dvalid, 'eval')],
                                  early_stopping_rounds=10, num_boost_round=500, xgb_model=trained_model)

    
    # Forecast using the model
    test_pred = trained_model.predict(dtest)

    # Prepare a DataFrame to collect results
    batch_results = pd.DataFrame({
        'permno': [permno.decode('utf-8') for permno in permnos],
        'date': [date.decode('utf-8') for date in dates],
        'label': y_test,
        'ME': me,
        'neg_ret': 1 - test_pred,
        'pos_ret': test_pred
    })
    
    # Append to the results list
    results.append(batch_results)
    
    
    # Only run this block the very first iteration
    if train_start == 199301:
        dtrain = xgb.DMatrix(Chars_train, label=Labels_train_class)
        
        # Forecast using the model
        train_pred = trained_model.predict(dtrain)
        
        # Extract training and validation data
        with h5py.File(hdf5_dataset_path, 'r') as file:
            permnos = file['permnos'][mask_train]
            dates = file['dates'][mask_train]
            me = file['ME'][mask_train]
        
        # Prepare a DataFrame to collect results
        batch_results_train = pd.DataFrame({
            'permno': [permno.decode('utf-8') for permno in permnos],
            'date': [date.decode('utf-8') for date in dates],
            'label': Labels_train,
            'ME': me,
            'neg_ret': 1 - train_pred,
            'pos_ret': train_pred
        })
    
    # Move the window one month forward
    if train_start % 100 != 12:
        train_start += test_window
    else:
        train_start += 100 - 11

    # return all_forecasts

df = pd.concat(results, ignore_index=True)

df.to_csv(f'{DATA_PATH}/returns/XGBoost_OOS.csv')
batch_results_train.to_csv(f'{DATA_PATH}/returns/XGBoost_IS.csv')

# Get the sharpe ratio
sharpe_ratio, annualized_returns, annualized_stds = compute_sharpe_ratio(batch_results_train, date_col = 'date', pred_col='pos_ret', ret_col='label', me_col='ME')






















