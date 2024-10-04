# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:30:25 2024

@author: bjark
"""


import sys
sys.path.append('C:/Users/bjark/Documents/AU/Kandidat/4. Semester/Code/Speciale-Code')

from project_imports import *

from importlib import reload
import generate_graphs
import functions

import copy


# Log into the Wharton database
conn = wrds.Connection(wrds_username='bjarki')

# Setting start and end date for the data
start_date = "01/01/1993"
end_date = "01/01/2022"


# Set initial parameters
start_year = 1993
initial_end_year = 1999
window_size = 4
overlap = 1
drop_rate = 0.0
adj_prc = False

# Reload the required modules
reload(functions)
reload(generate_graphs)

dates_to_predict = list(crspm.date.unique())

# Initialize an empty list to collect all outputs
dataset = []

# Loop over the years with a 5-year time window and 1-year overlap
while start_year + window_size - 1 <= initial_end_year or start_year < initial_end_year:
    # Calculate end year for the current window
    end_year = start_year + window_size - 1
    
    if end_year > initial_end_year:
        end_year = initial_end_year
        
    # Set the start and end dates for this iteration
    start_date = f"01/01/{start_year}"
    end_date = f"01/01/{end_year}"
    
    print(start_date, end_date)


####### 10516_1993-03-08.png' 10375_1994-11-28.png
######### ONLY RUN TO UPDATE EXISTING DATA #########

    # Reload the module.
    reload(functions)
    
    # # Labels to be predicted
    # labels = crspm[(crspm['date'] >= pd.to_datetime(start_date)) & (crspm['date'] <= pd.to_datetime(end_date))][['permno', 'date', 'excess_ret_ahead']]

    
    # sp500_daily = functions.load_sp500_data(conn, start_date, end_date, freq = 'daily', add_desc=False)
    US_market = functions.load_US_market(conn, start_date, end_date, freq = 'daily', add_desc=False, ret = False)
    
    US_market = US_market.rename(columns={
        'date': 'Date', 
        'prc': 'Close', 
        'openprc': 'Open', 
        'askhi': 'High', 
        'bidlo': 'Low', 
        'vol': 'Volume',
        'ret20': 'Ret20'
    })
    
    US_market.Close = US_market.Close.abs()
    
    # sp500_daily.to_csv(f'{DATA_PATH}/sp500_daily.csv', index=False)
    
    # Reload the module
    reload(generate_graphs)
    
    # Define the window size to be used (input)
    ws = 20
    
    first_trading_day = "01-01-2001"
    
    market = 'US'
    
    # Get out the dataset
    dataset_gen = generate_graphs.GraphDataset(df = US_market, win_size=ws, mode='train', label='Ret20', market = market,
                                           indicator = [{'MA': 20}], show_volume=True, drop_rate = drop_rate, adj_prc=adj_prc,
                                           predict_movement=True, dates_to_gen=dates_to_predict, parallel_num=-1)
    
    # Generate the image set
    dataset_append = dataset_gen.generate_images()
    
    # Append the generated output to `dataset`
    dataset.extend(dataset_append)  # Use `.extend()` to combine all lists into `dataset`
    
    # Move to the next start year (this creates the 1-year overlap)
    start_year += window_size - overlap - 1

# # Filter out permno's with less than 40 observations
# sp500_daily = sp500_daily.groupby('permno').filter(lambda x: len(x) >= 40)
# US_market = US_market.groupby('permno').filter(lambda x: len(x) >= 40)


# # Ensure 'date' column is of datetime type
# US_market['date'] = pd.to_datetime(US_market['date'])

# # Group by 'permno' and 'date' and check for multiple 'exchcd' values
# multiple_exchcd = US_market.groupby(['permno', 'date'])['exchcd'].nunique().reset_index()
# multiple_exchcd = multiple_exchcd[multiple_exchcd['exchcd'] > 1]

# # Find any permno with multiple observations on the same date
# multiple_observations = df.groupby(['permno', 'date']).size().reset_index(name='count')
# multiple_observations = multiple_observations[multiple_observations['count'] > 1]

# # Display the results
# print("Permno with multiple 'exchcd' values on the same date:")
# print(multiple_exchcd)

# print("\nPermno with multiple observations on the same date:")
# print(multiple_observations)

# # Aggregate unique exchange codes for each permno and count them
# result = US_market.groupby('permno').agg({
#     'exchcd': lambda x: set(x)  # Using set to get unique exchange codes
# })

# # Add a column to count the number of unique exchange codes
# result['exchcd_count'] = result['exchcd'].apply(len)

# # Filter to get only those permnos with more than one exchcd
# multiple_exchcd_permnos = result[result['exchcd_count'] > 1]

# # Print results
# print("Permno with multiple 'exchcd' values and their exchange codes:")
# print(multiple_exchcd_permnos[['exchcd', 'exchcd_count']])

# p = US_market[US_market.permno == 10051]

start_date = "01/01/2001"

US_market_monthly = functions.load_US_market(conn, start_date, end_date, freq = 'monthly', add_desc=False)
# sp500_monthly = functions.load_sp500_data(conn, start_date, end_date, freq = 'monthly')

ff5_monthly = functions.load_ff5_data(conn, start_date, end_date, freq = 'monthly')

conn.close()


sp500_daily = sp500_daily.rename(columns={
    'date': 'Date', 
    'prc': 'Close', 
    'openprc': 'Open', 
    'askhi': 'High', 
    'bidlo': 'Low', 
    'vol': 'Volume',
    'ret20': 'Ret20'
})

# sp500_daily.to_csv(f'{DATA_PATH}/sp500_daily.csv', index=False)


# # Create or open the HDF5 file
# with h5py.File(hdf5_train_path, 'a') as hdf5_file:
    
#     total_images = 6000000
    
#     # Create the dataset
#     images_dataset = hdf5_file.create_dataset("images", (total_images,), dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
#     labels_dataset = hdf5_file.create_dataset("labels", (total_images,), dtype=np.float32)
#     permnos_dataset = hdf5_file.create_dataset("permnos", (total_images,), dtype=h5py.string_dtype(encoding='utf-8'))
#     dates_dataset = hdf5_file.create_dataset("dates", (total_images,), dtype=h5py.string_dtype(encoding='utf-8'))


# train_data, test_data, table = dataset.generate_images()


#####################################################

# Load the CSV files into DataFrames

# Load the sp500 data
sp500_daily = pd.read_csv(f'{DATA_PATH}/sp500_daily.csv')

# Convert 'Date' column to datetime type if not already
sp500_daily['Date'] = pd.to_datetime(sp500_daily['Date'])

# sp500_monthly = pd.read_csv(f'{DATA_PATH}/sp500_monthly.csv')

# Load the image data
table = 'I20VolTInd20'
# train_data = pd.read_csv(f'{DATA_PATH}/{market}/{table}_dataset_train.csv')
# test_data = pd.read_csv(f'{DATA_PATH}/{market}/{table}_dataset_test.csv')

### Switch to use hdf5 file system
hdf5_train_path = f'{DATA_PATH}/{market}/{table}_train.h5'
hdf5_test_path = f'{DATA_PATH}/{market}/{table}_test.h5'



# # Extract date and convert to datetime
# train_data['date'] = pd.to_datetime(train_data.iloc[:,0].str.split('_').str[-1].str.replace('.png', ''), format='%Y-%m-%d')
# # train_data['permno'] = train_data.iloc[:,0].str.split('_').str[0]

# test_data['date'] = pd.to_datetime(test_data.iloc[:,0].str.split('_').str[-1].str.replace('.png', ''), format='%Y-%m-%d')
# test_data['permno'] = test_data.iloc[:,0].str.split('_').str[0]

# train_data = train_data.values.tolist()
# test_data = test_data.values.tolist()
# (entry, path, run=None)
generate_graphs.show_single_graph(hdf5_train_path, 140800)
generate_graphs.show_single_graph(hdf5_test_path, 569989)






#########################
##  Define the Models  ##
#########################

import custom_model
import custom_dataset

# Reload the modules
reload(custom_model)
reload(custom_dataset)

# scp "C:\Users\bjark\Documents\AU\Kandidat\4. Semester\Code\Speciale-Code\data\US\I20VolTInd20_train.h5" olsenbja@lumi.csc.fi:/scratch/project_465001092/Speciale-Code/data/US
# Load the dataset without transformations for now
graph_dataset_init = custom_dataset.GraphDataset(path=hdf5_train_path, transform=transforms.ToTensor(), mode='train')

init_loader = DataLoader(graph_dataset_init, batch_size=128, shuffle=False, num_workers=4)

# Initialize variables for calculating mean and std
mean = 0.0
std = 0.0
n_samples = 0

for batch in tqdm(init_loader, desc="Calculating mean and std"):
    images = batch['image']
    batch_samples = images.size(0)  # Batch size (number of images in the batch)
    images = images.view(batch_samples, images.size(1), -1)  # Flatten the images to (batch_size, channels, height*width)
    
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += batch_samples

mean /= n_samples
std /= n_samples

# Convert the computed mean and std from tensor to float
computed_mean = mean.item()
computed_std = std.item()

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

# Initialize the dataset
# graph_dataset_train = custom_dataset.GraphDataset(train_data, path = table, transform=custom_transforms, mode = 'train')
# graph_dataset_test = custom_dataset.GraphDataset(test_data, path = table, transform=custom_transforms, mode = 'test')

# Initialize your dataset
graph_dataset_train = custom_dataset.GraphDataset(path = hdf5_train_path, transform=custom_transforms, mode = 'train')
graph_dataset_test = custom_dataset.GraphDataset(path = hdf5_test_path, transform=custom_transforms, mode = 'test')


# Define some variables
train_ratio = 0.8
num_in_batch = 128
num_workers = 8
reg_label = False


# Create the dataloaders
train_loader_size = int(len(graph_dataset_train)*train_ratio)
valid_loader_size = len(graph_dataset_train) - train_loader_size


train_loader, valid_loader = random_split(graph_dataset_train, [train_loader_size, valid_loader_size])

train_loader = DataLoader(dataset=train_loader, batch_size=num_in_batch, 
                          shuffle=True, num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(dataset=valid_loader, batch_size=num_in_batch, 
                          shuffle=True, num_workers=num_workers, pin_memory=True)


# DataLoader
test_loader = DataLoader(graph_dataset_test, batch_size=num_in_batch, 
                         shuffle=False, num_workers=num_workers, pin_memory=True)



def check_for_missing_data(dataloader, pred_win):
    """
    Iterates through the DataLoader to check for NaN values in inputs and labels.
    """
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Checking data")):
        inputs = batch['image']
        labels = batch[f'ret{pred_win}']
        
        # Check for NaNs in inputs
        if torch.isnan(inputs).any():
            print(f"NaN detected in inputs at batch {batch_idx}")
        
        # Check for NaNs in labels
        if torch.isnan(labels).any():
            print(f"NaN detected in labels at batch {batch_idx}")
        
        # Optionally: Check for infinity
        if torch.isinf(inputs).any():
            print(f"Infinity detected in inputs at batch {batch_idx}")
        if torch.isinf(labels).any():
            print(f"Infinity detected in labels at batch {batch_idx}")

# Use the function to check for NaNs in the training and validation loaders
print("Checking training data for NaNs or missing data...")
check_for_missing_data(train_loader, ps)

print("Checking validation data for NaNs or missing data...")
check_for_missing_data(valid_loader, ps)


# for i, batch in enumerate(train_loader):
#     print(batch)
#     if i == 0:
#         break

# Start the training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate the model
CNN2D_model = custom_model.CNNModel(
    win_size=ws,
    inplanes=64,
    drop_prob=0.50,
    batch_norm=True,
    xavier=True,
    lrelu=True,
    bn_loc="bn_bf_relu",
    regression_label=reg_label
)

CNN2D_model.to(device)


n_epochs = 5
ps = 20

import train

# Reload the module
reload(train)

epoch_stats, best_validate_metrics, model = train.train_n_epochs(n_epochs = n_epochs, 
                                                                    model = CNN2D_model, 
                                                                    pred_win = ps, 
                                                                    train_loader = train_loader, 
                                                                    valid_loader = valid_loader, 
                                                                    early_stop = False, 
                                                                    early_stop_patience = 10, 
                                                                    lr=1e-4,
                                                                    regression_label=reg_label)

train.plot_epoch_stats(epoch_stats)

# Save trained model
# torch.save(model.state_dict(), f'{PROJECT_PATH}/model/100101_231231.pth')
# torch.save(model.state_dict(), f'{PROJECT_PATH}/model/930101_200101.pth')
# torch.save(model.state_dict(), f'{PROJECT_PATH}/model/NYSE_930101_200101.pth')
torch.save(model.state_dict(), f'{PROJECT_PATH}/model/h5_2_NYSE_930101_200101.pth')

################# Predict ################
# Load the saved model state
model = CNN2D_model  # Initialize the model
# model.load_state_dict(torch.load(f'{PROJECT_PATH}/model/100101_231231.pth'))
# model.load_state_dict(torch.load(f'{PROJECT_PATH}/model/NYSE_930101_200101.pth'))
model.load_state_dict(torch.load(f'{PROJECT_PATH}/model/LUMI_Train_NYSE_930101_200101.pth'))

model.to(device)
model.eval()  # Set the model to evaluation mode


# import torch



# Assume the following:
# dataloader: the DataLoader object providing the batch data
# model_list: list of models that should be evaluated
# device: the device (CPU or GPU) on which the models are evaluated

# # DataFrame column names and data types
# df_columns = ["permno", "date", "pos_prob"]
# df_dtypes = {'permno': object, "date": "datetime64[ns]", "pos_prob": np.float64}

# # Initialize list to collect DataFrames
# results = []

# # Run inference
# for batch in test_loader:
#     image = batch["image"].to(device, dtype=torch.float)
#     total_prob = torch.zeros(len(image), 2 if reg_label is False else 1, device=device)
    
#     # Process each model
#     with torch.no_grad():
#         outputs = model(image)
#         if reg_label is False:
#             outputs = nn.Softmax(dim=1)(outputs)
#         total_prob += outputs
    
#     # Create empty DataFrame for the batch
#     batch_df = pd.DataFrame(index=range(len(image)), columns=df_columns)
#     batch_df = batch_df.astype(df_dtypes)
    
#     # Fill the DataFrame
#     batch_df["permno"] = batch["permno"]
#     batch_df["date"] = pd.to_datetime(batch["date"])
#     # batch_df["ret_val"] = np.nan_to_num(batch["ret20"].numpy())
    
#     if reg_label is False:
#         batch_df["pos_prob"] = total_prob[:, 1].cpu().numpy()
#     else:
#         batch_df["pos_prob"] = total_prob.squeeze().cpu().numpy()

#     results.append(batch_df)

# # Concatenate all batch DataFrames
# df = pd.concat(results, ignore_index=True)
# df.reset_index(drop=True, inplace=True)

# # Output DataFrame
# print(df.head())

# results_df = pd.DataFrame(df)


#############

# Initialize list to collect DataFrames
results = []

# Disable gradient computation for inference
with torch.no_grad():
    # Wrap your data loader with tqdm for a progress bar
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Predicting'):
        images = batch['image'].to(device)  # Ensure images are on the same device as the model
        outputs = model(images)  # Get logits from the model

        # Record logits along with permno and date for each item in the batch
        for j in range(images.size(0)):
            results.append({
                'permno': batch['permno'][j],
                'date': batch['date'][j],
                'neg_ret': outputs[j, 0].item(),  # Logit for negative return
                'pos_ret': outputs[j, 1].item()   # Logit for positive return
            })

# Convert results list to a DataFrame
results_df = pd.DataFrame(results)

# Optionally save to CSV
# results_df.to_csv('logits_output.csv', index=False)

#################

# Convert 'date' to datetime to ease extraction of year and month
results_df['date'] = pd.to_datetime(results_df['date'])

# Apply softmax to the logits for each month separately
def apply_softmax(group):
    logits = torch.tensor(group[['neg_ret', 'pos_ret']].values)
    probabilities = nn.functional.softmax(logits, dim=1).numpy()  # Applying softmax across each row
    group['pos_prob'] = probabilities[:, 1]  # Store only the probability of positive return
    return group

# Group by year and month, and apply softmax
results_df = results_df.groupby([results_df['date'].dt.year, results_df['date'].dt.month]).apply(apply_softmax)



# Function to assign quintiles
def assign_quintiles(group):
    group['quintile'] = pd.qcut(group['pos_prob'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return group

# Apply function to each monthly group
results_df = results_df.groupby([results_df['date'].dt.year, results_df['date'].dt.month]).apply(assign_quintiles)

# Reset the index to make sure 'date' is treated as a column
results_df.reset_index(drop=True, inplace=True)

# Now sort by 'date' and 'quintile'
results_df.sort_values(by=['date', 'pos_prob'], inplace=True)


# Convert 'permno' to string in both DataFrames if they are supposed to be treated as categorical identifiers
# sp500_monthly['permno'] = sp500_monthly['permno'].astype(str)

US_market_monthly['permno'] = US_market_monthly['permno'].astype(str)

results_df['permno'] = results_df['permno'].astype(str)


# Merge the two DataFrames on 'permno' and 'date'
merged_df = pd.merge(US_market_monthly, results_df, on=['permno', 'date'])


# Select the necessary columns
merged_df = merged_df[['date', 'permno', 'prc', 'ret', 'quintile']]

def shift_returns(df, cols):
    # Ensure the dataframe is sorted by permno and date to maintain the correct order for shifting
    df = df.sort_values(by=['permno', 'date'])
    
    # Shift the 'ret' column by -1 within each group defined by 'permno'
    # This will move each return one month ahead within each stock group
    for col in cols:
        df[f'{col}_1m_ahead'] = df.groupby('permno')[col].shift(-1)
    
    return df


# Ensure date columns are datetime objects
merged_df['date'] = pd.to_datetime(merged_df['date'])

ff5_indexed = ff5_monthly.copy()
ff5_indexed['dateff'] = pd.to_datetime(ff5_indexed['dateff'])

# Set date as index for easy subtraction
merged_df.set_index('date', inplace=True)
ff5_indexed.set_index('dateff', inplace=True)

# Use join to merge on indices
merged_df = merged_df.join(ff5_indexed[['rf']], how='left')

# Reset index if needed to bring date back as a column
merged_df.reset_index(inplace=True)

# Assuming merged_df is your dataframe
merged_df = shift_returns(merged_df, ['ret', 'rf'])

# Calculate excess returns using the correctly aligned risk-free rate
merged_df['excess_ret'] = merged_df['ret_1m_ahead'] - merged_df['rf_1m_ahead']




# ############ TEST
# # Apply function to each monthly group
# tester = merged_df.dropna()
# merged_df = tester.groupby([tester['date'].dt.year, tester['date'].dt.month]).apply(assign_quintiles)

# # Reset the index to make sure 'date' is treated as a column
# merged_df.reset_index(drop=True, inplace=True)
# ############

# def calculate_portfolio_metrics(df):
    
#     # Group by 'quintile' and calculate monthly excess return and standard deviation
#     monthly_means = df.groupby(['date', 'quintile'])['excess_ret'].mean().unstack()
#     monthly_stds = df.groupby(['date', 'quintile'])['excess_ret'].std().unstack()
    
#     # Rename 'Q10' to 'Low' and 'Q1' to 'High'
#     monthly_means.rename(columns={10: 'High', 1: 'Low'}, inplace=True)
#     monthly_stds.rename(columns={10: 'High', 1: 'Low'}, inplace=True)
    
#     # Calculate High-Low (H-L) monthly returns and standard deviation
#     monthly_means['H-L'] = monthly_means['High'] - monthly_means['Low']
    
#     # Calculate covariance between High and Low returns
#     cov_hl = df.pivot_table(index='date', columns='quintile', values='excess_ret').cov().loc[1, 10]
    
#     # Calculate variance for H-L portfolio
#     variance_hl = ((monthly_stds['High'] ** 2) + (monthly_stds['Low'] ** 2)) - 2 * cov_hl
#     monthly_stds['H-L'] = np.sqrt(variance_hl)

#     return monthly_means, monthly_stds

def calculate_portfolio_returns(df):
    
    # Pivot the DataFrame to have quintile portfolios as columns
    monthly_means = df.pivot_table(index='date', columns='quintile', values='excess_ret', aggfunc='mean')

    # Calculate High-Low (H-L) monthly returns
    monthly_means['H-L'] = monthly_means[10] - monthly_means[1]
    
    # Rename 'Q10' to 'Low' and 'Q1' to 'High'
    monthly_means.rename(columns={10: 'High', 1: 'Low'}, inplace=True)

    return monthly_means

def calculate_annualized_metrics(monthly_returns):
    # Annualize the returns and standard deviations
    annualized_returns = monthly_returns.mean() * 12
    annualized_stds = monthly_returns.std() * np.sqrt(12)

    # Calculate the Sharpe ratios
    sharpe_ratios = annualized_returns / annualized_stds

    return annualized_returns, annualized_stds, sharpe_ratios


# Get the monthly returns and std for each portfolio and add the H-L portfolio
monthly_returns = calculate_portfolio_returns(merged_df)

# Calculate annualized returns, standard deviations, and Sharpe ratios
annualized_returns, annualized_stds, sharpe_ratios = calculate_annualized_metrics(monthly_returns)

# Output the results
print("Annualized Returns:\n", annualized_returns)
print("\nAnnualized Standard Deviations:\n", annualized_stds)
print("\nSharpe Ratios:\n", sharpe_ratios)



# Group by 'quintile' and calculate monthly excess return and standard deviation
monthly_means_US_market = merged_df.groupby(['date'])['excess_ret'].mean().rename('Mkt')

# Shift the returns down by one month within each quintile
adjusted_monthly_returns = monthly_returns.shift(1)

# adjusted_monthly_returns = adjusted_monthly_returns.join(ff5_indexed[['mktrf']], how='left')
adjusted_monthly_returns = adjusted_monthly_returns.join(monthly_means_US_market, how='left')

# Set the first row to 1 for each quintile to normalize the start point
adjusted_monthly_returns.iloc[0] = 0







# Get the cumulative log returns
cumulative_returns = np.log((1 + adjusted_monthly_returns)).cumsum()
# cumulative_returns = (1 + adjusted_monthly_returns).cumprod() - 1
# cumulative_returns = (adjusted_monthly_returns).cumsum()

# Plotting each quintile's cumulative returns over time
plt.figure(figsize=(14, 8))

# Define the columns that should have dotted lines
dotted_columns = {'High', 'H-L', 'Mkt'}

for quintile in cumulative_returns.columns:
    # Check if the current column is in the dotted columns set
    if quintile in dotted_columns:
        plt.plot(cumulative_returns.index, cumulative_returns[quintile], linestyle='--', label=f'{quintile}')  # Dotted line
    else:
        plt.plot(cumulative_returns.index, cumulative_returns[quintile], label=f'{quintile}')  # Solid line

plt.title('Returns by Portfolio Quintile')
plt.xlabel('Date')
plt.ylabel('Log Cumulative Return')
plt.legend(title='Decile')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.tight_layout()  # Adjust plot to fit labels
plt.show()




















































def read_image_from_hdf5(hdf5_file, index):
    # Access the stored PNG byte data
    image_data = hdf5_file['images'][index]
    
    # Decode the image data from PNG
    img = Image.open(io.BytesIO(image_data)).convert('L')
    return img

# Usage:
with h5py.File(hdf5_train_path, 'r') as hdf5_file:
    img = read_image_from_hdf5(hdf5_file, 0)
    img.show()









def store_images_in_hdf5(train_csv_path, test_csv_path, hdf5_train_path, hdf5_test_path):
    # Process and store data in a specific HDF5 file
    def process_data(data, hdf5_path, mode):
        with h5py.File(hdf5_path, 'w') as hdf5_file:
            # Assume image size from your dataset loader example
            image_size = (64, 60)  # Adjust as needed
            total_images = len(data)
            
            compression_algo = ''  # or 'lzf' which is faster but less compressive
            compression_opts = 1  # Lower compression level for gzip
            
            # Create datasets in the HDF5 file with gzip compression
            # images_dataset = hdf5_file.create_dataset("images", (total_images,) + image_size, dtype=np.uint8, compression=compression_algo, compression_opts=compression_opts)
            # labels_dataset = hdf5_file.create_dataset("labels", (total_images,), dtype=np.float32, compression=compression_algo, compression_opts=compression_opts)
            # permnos_dataset = hdf5_file.create_dataset("permnos", (total_images,), dtype=h5py.string_dtype(encoding='utf-8'), compression=compression_algo, compression_opts=compression_opts)
            # dates_dataset = hdf5_file.create_dataset("dates", (total_images,), dtype=h5py.string_dtype(encoding='utf-8'), compression=compression_algo, compression_opts=compression_opts)
            
            images_dataset = hdf5_file.create_dataset("images", (total_images,), dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
            labels_dataset = hdf5_file.create_dataset("labels", (total_images,), dtype=np.float32)
            permnos_dataset = hdf5_file.create_dataset("permnos", (total_images,), dtype=h5py.string_dtype(encoding='utf-8'))
            dates_dataset = hdf5_file.create_dataset("dates", (total_images,), dtype=h5py.string_dtype(encoding='utf-8'))

            for i in tqdm(range(len(data)), desc=f'Processing {mode} data'):
                row = data.loc[i]
                file_path = os.path.join(f'{DATA_PATH}/{table}/', row['file_name'])
                
                # Extract permno and date from filename
                parts = row['file_name'].split('/')
                year_folder, filename = parts[0], parts[1]
                permno, date_part = filename.split('_')
                date = date_part.split('.')[0]  # Remove file extension
                
                try:
                    with open(file_path, 'rb') as img:
                        # image_array = np.array(img.convert('L'), dtype=np.uint8)
                        binary_data = img.read()
                        
                        # # Check if image needs to be resized
                        # if image_array.shape != image_size:
                        #     img = img.resize(image_size, Image.ANTIALIAS)
                        #     image_array = np.array(img, dtype=np.uint8)
                        
                    binary_data_np = np.asarray(binary_data)
                    binary_blob = np.void(binary_data)
                    
                    images_dataset[i] = np.frombuffer(binary_data, dtype='uint8')
                    labels_dataset[i] = row['ret']
                    permnos_dataset[i] = permno
                    dates_dataset[i] = date
                        
                except Exception as e:
                    print(f'Error processing image {file_path}: {str(e)}')

            print(f"All {mode} images and labels have been stored to HDF5.")

    # Load CSV data
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    # Process training and test data
    process_data(train_data, hdf5_train_path, 'Train')
    process_data(test_data, hdf5_test_path, 'Test')

# Define the paths
table = 'I20VolTInd20'
# market = 'your_market_directory'  # Ensure to define or update your market directory
# DATA_PATH = 'your_data_directory_path'  # Define your DATA_PATH if not already defined

train_csv_path = f'{DATA_PATH}/{market}/{table}_dataset_train.csv'
test_csv_path = f'{DATA_PATH}/{market}/{table}_dataset_test.csv'
hdf5_train_path = f'{DATA_PATH}/{market}/I20VolTInd20_train.h5'
hdf5_test_path = f'{DATA_PATH}/{market}/I20VolTInd20_test.h5'

# Function call
store_images_in_hdf5(train_csv_path, test_csv_path, hdf5_train_path, hdf5_test_path)



 # with open('C:/Users/bjark/Documents/AU/Kandidat/4. Semester/Code/Speciale-Code/data/I20VolTInd20/1993/10057_1993-03-02.png', 'rb') as img:
 #     # image_array = np.array(img.convert('L'), dtype=np.uint8)
 #     binary_data = img.read()



import h5py
import numpy as np
import matplotlib.pyplot as plt
import io

def plot_images_from_hdf5(hdf5_test_path, num_images=10):
    # Open the HDF5 file in read mode
    with h5py.File(hdf5_test_path, 'r') as hdf5_file:
        # Extract the images dataset
        images_dataset = hdf5_file['images']
        
        # Plot the first `num_images` images
        plt.figure(figsize=(15, 15))
        
        for i in range(num_images):
            # Decode the image from its binary form
            binary_image = images_dataset[i]
            image_array = np.frombuffer(binary_image, dtype=np.uint8)
            
            # Reshape the image to the original dimensions (e.g., 64x60) 
            # (You may need to adjust the shape based on how the images are stored)
            image = image_array.reshape((64, 60))  # Adjust this if necessary
            
            # Plot each image in a subplot
            plt.subplot(1, num_images, i+1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        
        plt.show()

# Define the path to the HDF5 test file
# hdf5_test_path = f'{DATA_PATH}/{market}/I20VolTInd20_test.h5'

# Call the function to plot images
plot_images_from_hdf5(hdf5_test_path, num_images=10)





import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

def show_single_graph_from_h5(hdf5_file, run=None):
    # Access the images and other datasets from the hdf5 file
    images_dataset = hdf5_file['images']
    labels_dataset = hdf5_file['labels']
    dates_dataset = hdf5_file['dates']
    permnos_dataset = hdf5_file['permnos']
    
    total_images = len(images_dataset)  # Total number of images

    # Loop through all images with tqdm to show progress
    for entry_idx in tqdm(range(total_images), desc="Plotting Images"):
    
        # Extract the binary image data
        binary_image = images_dataset[entry_idx]
        
        # Convert the binary data to a NumPy array
        # image_array = np.frombuffer(binary_image, dtype=np.uint8)
        image = Image.open(io.BytesIO(binary_image))
        
        # Assuming the image is grayscale with known dimensions (e.g., 64x60)
        # image_size = (64, 60)  # Adjust as needed
        # image_array = image_array.reshape(image_size)
        
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Retrieve other metadata for the plot
        label = labels_dataset[entry_idx].round(3)
        date = dates_dataset[entry_idx].decode('utf-8')
        permno = permnos_dataset[entry_idx].decode('utf-8')
        
        # Plot the image
        plt.figure()
        plt.imshow(image_array, cmap=plt.get_cmap('gray'))
        plt.title(f'Ret20: {label:.3f}\n Last Date is: {date}')
        plt.axis('off')  # Optional: Hide the axes
        plt.tight_layout()
    
        if run is not None:
            # Save the plot to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                plt.savefig(tmpfile.name, format='png')
            
            # Log the plot to Neptune
            # run["plots/show_single_graph"].upload(tmpfile.name)
            plt.close(plt.gcf())
        else:
            plt.show()

# Usage Example:
# Assuming you have already opened the HDF5 file
with h5py.File(hdf5_train_path, 'r') as hdf5_file:
    # Example of plotting the image at index 0
    show_single_graph_from_h5(hdf5_file)















# Open the HDF5 file
with h5py.File(hdf5_test_path, 'r') as hdf5_file:
    # List all the datasets and groups in the file
    def print_structure(name, obj):
        print(name)
    
    # Print the structure of the HDF5 file
    hdf5_file.visititems(print_structure)


# Check if the file exists
if os.path.exists(hdf5_test_path):
    print("File exists")
else:
    print("File does not exist")

import os

# Check the file size
file_size = os.path.getsize(hdf5_test_path)
print(f"File size: {file_size} bytes")





import custom_model
import custom_dataset

# Reload the modules
reload(custom_model)
reload(custom_dataset)

from torch.utils.data import DataLoader

# Create the dataset
train_dataset = custom_dataset.GraphDataset(hdf5_path=hdf5_test_path, transform=custom_transforms, mode='train')

# Create the dataloader (can specify num_workers > 0 for multiple workers)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=14, pin_memory=True)

# Add a tqdm progress bar to track the iteration over the dataset
for batch in tqdm(train_loader, desc="Training Progress", unit="batch"):
    images = batch['image']
    labels = batch['ret20']
    # Your training code here




a = hdf5_file["images"][:][pd.isna(hdf5_file["images"][:])]
b = hdf5_file["labels"][:][pd.isna(hdf5_file["labels"][:])]
c = hdf5_file["permnos"][:][pd.isna(hdf5_file["permnos"][:])]
d = hdf5_file["dates"][:][pd.isna(hdf5_file["dates"][:])]









