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
##  Run XGB on US Market stocks  ##
###################################

model_run_nr = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

# Specify the data path
table = 'I20VolTInd20'
market = 'US'
ws = 20

# Setup the path to the dataset
hdf5_dataset_path = f'{DATA_PATH}/{market}/{table}_dataset.h5'

# Initialize the model (use the actual class reference)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


# def portfolio_sorting(df, pred_col='pred', ret_col='ret', me_col=None, n_portfolios=10):
#     """
#     General function to sort a DataFrame into portfolios based on predictions, with support for weighted sorting.

#     Args:
#         df (pd.DataFrame): DataFrame containing the data.
#         pred_col (str): Column name of the predictions used for sorting (default is 'pred').
#         ret_col (str): Column name of the actual returns to calculate portfolio returns (default is 'ret').
#         me_col (str, optional): Column name of the market equity weights (default is None for equal weighting).
#         n_portfolios (int): Number of portfolios to sort into (default is 10 for deciles).

#     Returns:
#         pd.DataFrame: DataFrame with sorted portfolios and associated returns.
#     """
#     # Ensure input DataFrame contains the necessary columns
#     if pred_col not in df.columns or ret_col not in df.columns:
#         raise ValueError(f"Columns '{pred_col}' and '{ret_col}' must be present in the input DataFrame.")
    
#     # Create a new column for portfolio allocation based on predictions
#     df = df.copy()  # Avoid modifying the original DataFrame

#     # Handle weighting (optional)
#     if me_col is not None:
#         # Use weighted quantile binning if market equity (ME) weights are provided
#         df['weight'] = df[me_col] / df[me_col].sum()
#     else:
#         # Use equal weights if no weights are provided
#         df['weight'] = 1.0 / len(df)

#     # Sort the DataFrame by predictions
#     df.sort_values(by=pred_col, inplace=True)

#     # Compute the cumulative sum of the weights to get weighted quantiles
#     df['cum_weight'] = df['weight'].cumsum()

#     # Determine the breakpoints for portfolio allocation (soft inequalities)
#     # Use np.linspace to get the quantile cut points
#     cut_points = np.linspace(0, 1.000001, n_portfolios + 1)

#     # Assign portfolios based on cumulative weight with soft inequality handling
#     df['portfolio'] = pd.cut(df['cum_weight'], bins=cut_points, labels=range(1, n_portfolios + 1), include_lowest=True, right=False)
    
#     return df

# def assign_portfolio(data, sorting_variable, n_portfolios):
#     """Assign portfolios to a bin between breakpoints."""
    
#     breakpoints = np.quantile(
#       data[sorting_variable].dropna(), 
#       np.linspace(0, 1, n_portfolios + 1), 
#       method="linear"
#     )
    
#     assigned_portfolios = pd.cut(
#       data[sorting_variable],
#       bins=breakpoints,
#       labels=range(1, breakpoints.size),
#       include_lowest=True,
#       right=True
#     )
    
#     return assigned_portfolios



# # Define a function to compute the Sharpe Ratio
# def compute_sharpe_ratio(data, date_col, pred_col, ret_col, me_col, n_portfolios=10):
#     """
#     Calculate the Sharpe ratio given an array of portfolio returns.
    
#     Args:
#         data (array-like): Array of returns.
        
#     Returns:
#         float: Sharpe ratio.
#     """
    
#     # # Apply function to each monthly group
#     # results_df = data.groupby(data[date_col]).apply(portfolio_sorting, pred_col=pred_col, ret_col=ret_col, me_col=me_col, n_portfolios=10)
    
#     # # Reset the index to make sure 'date' is treated as a column
#     # results_df.reset_index(drop=True, inplace=True)
    
#     # # Now sort by 'date' and 'quintile'
#     # results_df.sort_values(by=[date_col, pred_col], inplace=True)

#     def portfolio_sorting(df):
#         df['portfolio'] = assign_portfolio(df, pred_col, n_portfolios)
#         return df

#     # Apply portfolio sorting to each month
#     results_df = data.groupby(date_col, group_keys=False).apply(portfolio_sorting)
    
#     # Calculate weighted returns for each portfolio
#     # monthly_means = results_df.groupby([date_col, 'portfolio'], observed=True).apply(
#     #     lambda x: np.sum(x[ret_col] * x['weight']) / x['weight'].sum()
#     # ).unstack()  # Unstack to get quintiles as columns

#     monthly_means = results_df.groupby([date_col, 'portfolio'])[ret_col].mean().unstack()

#     # Calculate High-Low (H-L) monthly returns using weighted means
#     monthly_means['H-L'] = monthly_means[10] - monthly_means[1]

#     # Rename quintiles for clarity
#     monthly_means.rename(columns={10: 'High', 1: 'Low'}, inplace=True)

#     # Annualize the returns and standard deviations (weighted standard deviations)
#     annualized_returns = monthly_means.mean() * 12
#     annualized_stds = monthly_means.std() * np.sqrt(12)

#     # Calculate the Sharpe ratios
#     sharpe_ratios = annualized_returns / annualized_stds
    
#     return sharpe_ratios, annualized_returns, annualized_stds, monthly_means

def portfolio_sort(data, date_col, pred_col, ret_col, me_col=None, n_portfolios=10):
    """
    Compute the Sharpe ratio by assigning returns directly to portfolios, allowing overlapping memberships.

    Args:
        data (pd.DataFrame): DataFrame containing stock returns and predictions.
        date_col (str): Column name for the date or month identifier (e.g., 'yyyymm').
        pred_col (str): Column name of the predictions used for sorting.
        ret_col (str): Column name of the actual returns.
        me_col (str): Column name for market equity weights (ignored in this implementation).
        n_portfolios (int): Number of portfolios to sort into (default is 10 for deciles).

    Returns:
        pd.DataFrame: DataFrame with mean returns for each portfolio and date.
    """
    # Create an empty list to store the results
    results = []

    # Group the data by the date_col (e.g., yyyymm)
    grouped = data.groupby(date_col)

    # Iterate through each monthly group
    for date, group in grouped:
        # Sort the group by the predictions
        group_sorted = group.sort_values(by=pred_col)

        # Calculate the quantile breakpoints
        quantiles = group_sorted[pred_col].quantile(np.linspace(0, 1, n_portfolios + 1))

        # Create an empty dictionary to store stocks in each portfolio
        portfolios = {i + 1: [] for i in range(n_portfolios)}

        # Assign stocks to portfolios based on quantile breakpoints, allowing overlap at breakpoints
        for i in range(n_portfolios):
            # Define the lower and upper bound for the portfolio
            lower_bound = quantiles.iloc[i]
            upper_bound = quantiles.iloc[i + 1]

            # Identify the stocks in the current portfolio range
            in_portfolio = group_sorted[(group_sorted[pred_col] >= lower_bound) & (group_sorted[pred_col] <= upper_bound)]

            # Assign stocks to the current portfolio
            portfolios[i + 1] = in_portfolio

        # Calculate the mean return for each portfolio
        for port, stocks in portfolios.items():
            mean_return = stocks[ret_col].mean()
            results.append({
                'date': date,
                'portfolio': port,
                'mean_return': mean_return
            })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results).pivot(index='date', columns='portfolio', values='mean_return')
    
    results_df['H-L'] = results_df[n_portfolios] - results_df[1]
    
    return results_df


# Define a function to compute the Sharpe Ratio
def compute_sharpe_ratio(data, date_col, pred_col, ret_col, me_col, n_portfolios=10):
    """
    Calculate the Sharpe ratio given an array of portfolio returns.
    
    Args:
        data (array-like): Array of returns.
        
    Returns:
        float: Sharpe ratio.
    """
    
    # # Apply function to each monthly group
    # results_df = data.groupby(data[date_col]).apply(portfolio_sorting, pred_col=pred_col, ret_col=ret_col, me_col=me_col, n_portfolios=10)
    
    # # Reset the index to make sure 'date' is treated as a column
    # results_df.reset_index(drop=True, inplace=True)
    
    # # Now sort by 'date' and 'quintile'
    # results_df.sort_values(by=[date_col, pred_col], inplace=True)

    

    # Apply portfolio sorting to each month
    monthly_means = portfolio_sort(data, date_col, pred_col, ret_col, me_col=None)
    
    # Calculate weighted returns for each portfolio
    # monthly_means = results_df.groupby([date_col, 'portfolio'], observed=True).apply(
    #     lambda x: np.sum(x[ret_col] * x['weight']) / x['weight'].sum()
    # ).unstack()  # Unstack to get quintiles as columns

    # monthly_means = results_df.groupby([date_col, 'portfolio'])[ret_col].mean().unstack()

    # # Calculate High-Low (H-L) monthly returns using weighted means
    # monthly_means['H-L'] = monthly_means[10] - monthly_means[1]

    # Rename quintiles for clarity
    monthly_means.rename(columns={n_portfolios: 'High', 1: 'Low'}, inplace=True)

    # Annualize the returns and standard deviations (weighted standard deviations)
    annualized_returns = monthly_means.mean() * 12
    annualized_stds = monthly_means.std() * np.sqrt(12)

    # Calculate the Sharpe ratios
    sharpe_ratios = annualized_returns / annualized_stds
    
    return sharpe_ratios, annualized_returns, annualized_stds, monthly_means



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
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
        'nthread': 8,  # Using 8 threads
        'tree_method': 'hist',  # Faster than 'exact', especially with larger datasets
        'seed': 42
    }
    
    # Add GPU parameters if GPU is available
    if device == 'cuda':
        params.update({
            'device': 'cuda',  # Use GPU for prediction
        })
        
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
        model = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)
        
        # Predict on validation set
        pred = model.predict(dvalid)
        
        # Combine predictions and actual returns into a DataFrame from the valid dataset, to be used for portfolio sort
        month_results = pd.DataFrame({'yyyymm': np.array(all_dates[mask_valid_hyp], dtype=int), 'ret': Labels_valid, 'pred': pred})
        
        # Append to the results list
        results.append(month_results)

        del X_train, X_valid, y_train, y_valid
        gc.collect()
        
        if valid_date_hyp_run_start % 100 == 12:
            valid_date_hyp_run_start = (valid_date_hyp_run_start - 12) + 101
            valid_date_hyp_run_end = (valid_date_hyp_run_end - 12) + 101
        else:
            valid_date_hyp_run_start += 1
            valid_date_hyp_run_end += 1
            
    results = pd.concat(results, ignore_index=True)
    
    # Get the sharpe ratio
    sharpe_ratio, _, _, _ = compute_sharpe_ratio(results, date_col = 'yyyymm', pred_col='pred', ret_col='ret', me_col=None)

    # Return negative Sharpe ratio of the H-L portfolio, for minimization
    return -sharpe_ratio['H-L']



# Function to save and load data
def save_data(data, filepath):
    if os.path.exists(filepath):
        # Load previous data
        with open(filepath, 'rb') as f:
            existing_data = pickle.load(f)
        # Merge new data
        data = pd.concat([existing_data, data], ignore_index=True)
    # Save updated data
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        return None



train_start = 199301 + (model_run_nr*100)

# train_start = 201301 - 700

train_window_years = 7
valid_window_years = 2
test_window = 1
train_ratio = 0.8
trained_model = None
best_param_value = np.inf

# # Track when next tuning period is
# next_tuning = train_start
# tuning_years = 1

# Load all dates
with h5py.File(hdf5_dataset_path, 'r') as file:
    all_dates = file['dates'][:]

train_end = train_start + train_window_years * 100

predict_year = train_end

# Define file paths for saving the results
results_folder = f'{DATA_PATH}/XGB_Stats/{predict_year // 100}'  # Folder for this specific year's predictions
os.makedirs(results_folder, exist_ok=True)

# File paths
best_params_path = os.path.join(results_folder, 'best_params.pkl')
shap_values_path = os.path.join(results_folder, 'shap_values.pkl')
batch_results_path = os.path.join(results_folder, 'XGBoost_OOS.csv')

# Load previous data if exists
results_df = pd.read_csv(batch_results_path) if os.path.exists(batch_results_path) else pd.DataFrame()

study = optuna.create_study(direction='minimize')

# Loop for hyperparameter tuning, training, and saving
with tqdm(total=((int(((predict_year+100) - train_end)/100)*12)), desc="Overall Progress", position=0) as pbar:
    while train_start + train_window_years * 100 < (predict_year+100):
        train_end = train_start + train_window_years * 100
        test_end = train_end + test_window
        print(f"Training from {train_start} to {train_end}, Testing from {train_end} to {test_end}")

        mask_train = (train_start <= np.array(all_dates, dtype=int)) & (np.array(all_dates, dtype=int) < train_end)
        with h5py.File(hdf5_dataset_path, 'r') as file:
            Chars_train = file['chars'][:][mask_train]
            Labels_train = file['labels'][mask_train]

        Labels_train_class = np.array([1 if label > 0 else 0 for label in Labels_train])

        # if train_start == next_tuning:
        
        study.optimize(lambda trial: objective(trial, Chars_train, Labels_train_class, Labels_train, 
                                        train_start, train_end, all_dates[mask_train], valid_window_years, device), 
                    n_trials=8, n_jobs=8, show_progress_bar=True)
        best_params = study.best_params
        
        # if study.best_value < best_param_value:
        best_params_global = study.best_params

        best_params_global.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'nthread': 8,  # Using 8 threads
                'tree_method': 'hist',  # Faster than 'exact', especially with larger datasets
            })
        
        # best_param_value = study.best_value
        trained_model = None

        # Save best params
        save_data(pd.DataFrame([best_params]), best_params_path)

        # next_tuning += tuning_years * 100

        X_train_size = int(len(Labels_train_class) * train_ratio)
        X_valid_size = len(Labels_train_class) - X_train_size
        X_train, X_valid, y_train, y_valid = train_test_split(Chars_train, Labels_train_class, test_size=X_valid_size, shuffle=True, random_state=42)

        mask_test = (train_end <= np.array(all_dates, dtype=int)) & (np.array(all_dates, dtype=int) < test_end)
        with h5py.File(hdf5_dataset_path, 'r') as file:
            X_test = file['chars'][:][mask_test]
            y_test = file['labels'][mask_test]
            permnos = file['permnos'][mask_test]
            dates = file['dates'][mask_test]
            me = file['ME'][mask_test]

        y_test_class = np.array([1 if label > 0 else 0 for label in y_test])
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test, label=y_test_class)

        # if trained_model is None:
        #     trained_model = xgb.train(best_params_global, dtrain, evals=[(dtrain, 'train'), (dvalid, 'eval')],
        #                             early_stopping_rounds=10, num_boost_round=2000)
        # else:
        #     trained_model = xgb.train(best_params_global, dtrain, evals=[(dtrain, 'train'), (dvalid, 'eval')],
        #                             early_stopping_rounds=10, num_boost_round=500, xgb_model=trained_model)

        trained_model = xgb.train(best_params_global, dtrain, evals=[(dtrain, 'train'), (dvalid, 'eval')],
                                    early_stopping_rounds=10, num_boost_round=100)

        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer.shap_values(X_test)

        # Save SHAP values
        save_data(pd.DataFrame(shap_values), shap_values_path)

        test_pred = trained_model.predict(dtest)

        batch_results = pd.DataFrame({
            'permno': [permno.decode('utf-8') for permno in permnos],
            'date': [date.decode('utf-8') for date in dates],
            'label': y_test,
            'ME': me,
            'neg_ret': 1 - test_pred,
            'pos_ret': test_pred
        })

        # Append results and save to CSV
        results_df = pd.concat([results_df, batch_results], ignore_index=True)
        results_df.to_csv(batch_results_path, index=False)

         # Cleanup after training is done for this time period
        print("Cleaning up memory after training and prediction phase...")

        print_memory_stats(epoch='After training')

        # Delete large variables no longer in use
        del X_train, X_valid, X_test, y_train, y_valid, y_test_class, dtrain, dvalid, dtest
        del Chars_train, Labels_train, shap_values
        torch.cuda.empty_cache()
        gc.collect()

        print_memory_stats(epoch='After training and cleaning')

        # Move window forward
        if train_start % 100 != 12:
            train_start += test_window
        else:
            train_start += 100 - 11

        pbar.update(1)
        run["training/progress"].log(pbar.n)

if model_run_nr == 0:

    # Only run this block at the very end
    train_start = 199301 # Set back to the start of the period
    train_end = train_start + train_window_years * 100
    test_end = train_end + test_window

    mask_train = (train_start <= np.array(all_dates, dtype=int)) & (np.array(all_dates, dtype=int) < train_end)
    # Extract training and validation data
    with h5py.File(hdf5_dataset_path, 'r') as file:
        Chars_train = file['chars'][:][mask_train]
        Labels_train = file['labels'][mask_train]
        
    # Convert labels to binary classes
    Labels_train_class = np.array([1 if label > 0 else 0 for label in Labels_train])

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

    # Collct In-Sample results
    batch_results_train.to_csv(f'{DATA_PATH}/XGB_Stats/XGBoost_IS.csv')


# # Collect OOS results
# df = pd.concat(results, ignore_index=True)
# df.to_csv(f'{DATA_PATH}/returns/XGBoost_OOS.csv')


# # Get the sharpe ratio
# sharpe_ratio, annualized_returns, annualized_stds = compute_sharpe_ratio(batch_results_train, date_col = 'date', pred_col='pos_ret', ret_col='label', me_col='ME')

# Stop the monitoring on Neptune
run.stop()
