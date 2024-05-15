
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
end_date = "12/31/2022"


# Reload the module
reload(functions)

sp500 = functions.load_sp500_data(conn, start_date, end_date, freq = 'monthly')


# sp500_long = conn.raw_sql(f"""
#                             SELECT b.permno, b.hexcd, b.date
#                             FROM crsp.dsf AS b
#                             WHERE b.hexcd IN (1,2,3)
#                             AND b.date >= '{start_date}'
#                             AND b.date <= '{end_date}'
#                             ORDER BY date;
#                             """, date_cols=['date'])

# Get a list of all companies
permno_list = sp500.permno.unique()



permno_list_long = permno_list.copy()


a = permno_list_long[:10000]

a = np.unique(np.concatenate((permno_list_long, permno_list)))

# Load the firm characteristics
# dfChar = functions.load_firm_char_data()
# dfChar = functions.load_firm_char_data(permno_list)
# dfChar = functions.load_firm_char_data(199212, 202301, a)
dfChar = pd.read_csv(f'{DATA_PATH}/firm_characteristics/combined_df.csv')

# Start from 1993 as the CNN data is first all available there
dfChar = dfChar[dfChar['yyyymm']>199612]
dfChar = dfChar[dfChar['yyyymm']<202101]


# Reload the module
reload(functions)

ff5 = functions.load_ff5_data(conn, start_date, end_date)


########################################
##  Test run XGBoost on fewer stocks  ##
########################################

######
# Preprocess of data
######


# See how many characteristics have an empty value
pd.set_option('display.max_rows',None)
dfChar.isna().sum().sort_values(ascending = False)

# Rank and scale the characteristics to have values ranging from -1 to 1, excluding 'yyyymm' and 'permno'
df = dfChar.set_index(['yyyymm', 'permno']).groupby('yyyymm').rank(pct=True) * 2 - 1

# Transform the characteristics to have the cross-sectional median value if a missing value
df = df.fillna(df.groupby('yyyymm').transform('median'))

# Reset index to bring 'yyyymm' and 'permno' back as columns
df = df.reset_index()

# Only keep sp500 companies
df = df[df.permno.isin(permno_list)]

# See how many characteristics have an empty value
df.isna().sum().sort_values(ascending = False)



# Drop the variables with more than 25% missing values
# limitPer = len(df) * .9
df = df.dropna(axis=1)

# See how many characteristics have an empty value
df.isna().sum().sort_values(ascending = False)

len(df.permno.unique())

########
# Start training process
########

# # Add the returns onto the companies
# df['ret'] = df[df.groupby('permno') == sp500.permno]

# Convert date to same type
sp500['yyyymm'] = sp500['date'].dt.strftime('%Y%m').astype(int)
sp500['start_yyyymm'] = sp500['start'].dt.strftime('%Y%m').astype(int)
sp500['ending_yyyymm'] = sp500['ending'].dt.strftime('%Y%m').astype(int)


# Perform the merge to include only the 'ret' column from 'sp500' into 'df'
merged_df = pd.merge(df, sp500[['yyyymm', 'permno', 'ret', 'start_yyyymm', 'ending_yyyymm']], on=['yyyymm', 'permno'], how='left')

# Filter to keep rows where 'yyyymm' falls within the 'start_yyyymm' and 'ending_yyyymm' period
filtered_df = merged_df[(merged_df['yyyymm'] > merged_df['start_yyyymm']) & (merged_df['yyyymm'] <= merged_df['ending_yyyymm'])]

# See how many characteristics have an empty value
filtered_df.isna().sum().sort_values(ascending = True)


# Filter rows where any column has NaN values
a = filtered_df[filtered_df.isna().any(axis=1)]


########
# Initialize training and valid dataset
########

# Choose some cutoff period
dfTrain = filtered_df[filtered_df.yyyymm < 200001]
dfTest = filtered_df[filtered_df.yyyymm >= 200001]

# Define the columns to exclude from the features
exclude_columns = ['yyyymm', 'permno', 'start_yyyymm', 'ending_yyyymm', 'ret']

# Separate features and target for the training data
X = dfTrain.drop(columns=exclude_columns + ['ret'])
y = dfTrain['ret']

# Split the training data into training and validation sets
train_ratio = 0.8
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=train_ratio, random_state=42)



import xgboost as xgb

# Convert the pandas DataFrames into DMatrix objects; XGBoost's optimized data structure
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# Specify your training parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',  # Use 'reg:linear' for older versions of XGBoost
    'eval_metric': 'rmse'
}

# Train the model
evallist = [(dvalid, 'eval'), (dtrain, 'train')]
num_round = 100
bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)

# bst now contains your trained model


import xgboost as xgb

# Print XGBoost build configuration
print(xgb.get_config())



try:
    # Create a small DMatrix
    dtrain = xgb.DMatrix([[1, 2], [2, 3]], label=[1, 2])
    
    # Specify parameters with 'gpu_hist', which requires GPU support
    params = {'tree_method': 'hist', 'device': 'cuda'}
    
    # Attempt to train a model using 'gpu_hist'
    bst = xgb.train(params, dtrain, num_boost_round=1)
    
    print("GPU support is enabled in your XGBoost installation.")
except xgb.core.XGBoostError as e:
    print("GPU support is not enabled in your XGBoost installation.")
    print("Error:", e)




####################################
##  Test run CNN on fewer stocks  ##
####################################


df = sp500[sp500['permno'].isin(permno_list)].copy()


# dfApplMSFT.reset_index(drop=True)


# Only keep the columns we need
# df = dfApplMSFT[['date', 'prc', 'openprc', 'askhi', 'bidlo', 'vol', 'ret5', 'ret20', 'ret60']].copy()
# df = dfApplMSFT.copy()
# df = sp500.copy()

# Keep the permno
# permno = df['permno'].unique()

# Rename columns
# df['date', 'prc', 'openprc', 'askhi', 'bidlo', 'vol', 'ret5', 'ret20', 'ret60'] = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Ret5', 'Ret20', 'Ret60']

df = df.rename(columns={
    'date': 'Date', 
    'prc': 'Close', 
    'openprc': 'Open', 
    'askhi': 'High', 
    'bidlo': 'Low', 
    'vol': 'Volume', 
    'ret5': 'Ret5', 
    'ret20': 'Ret20', 
    'ret60': 'Ret60'
})

# # Only keep the columns we need
# df = sp500[['date', 'prc', 'openprc', 'askhi', 'bidlo', 'vol', 'ret5', 'ret20', 'ret60']].copy()

# # Keep the permno
# permno = sp500['permno'].unique()

# # Rename columns
# df.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Ret5', 'Ret20', 'Ret60']



# Reload the module
reload(generate_graphs)

# Define the window size to be used (input)
ws = 20

# Get out the dataset
dataset = generate_graphs.GraphDataset(df = df, win_size=ws, mode='train', label='Ret5', 
                                       indicator = [{'MA': 20}], show_volume=True, 
                                       predict_movement=True, parallel_num=-1)


# Generate the image set
image_set = dataset.generate_images()

# # Plot a graph
# generate_graphs.show_single_graph(image_set[0][0])


# Concatenate the dataset
dataset_all = []
for data in image_set:
    for row in data:
        dataset_all.append(row)
image_set = []

# for graph in dataset_all:
#     generate_graphs.show_single_graph(graph)

generate_graphs.show_single_graph(dataset_all[8])

# Get the ticker
unique_tickers = sp500[sp500['permno'] == dataset_all[1][0]]['ticker'].unique()





#########################
##  Define the Models  ##
#########################

import custom_model
import custom_dataset

# Reload the modules
reload(custom_model)
reload(custom_dataset)


# # Concatenate the dataset
# dataset_all = []
# for symbol_data in image_set:
#     dataset_all = dataset_all + symbol_data[1]
# image_set = [] # clear memory



# Define transformations
custom_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
])

# Initialize your dataset
graph_dataset = custom_dataset.GraphDataset(dataset_all, transform=custom_transforms)

# Define some variables
train_ratio = 0.8
num_in_batch = 128
num_workers = 1
reg_label = False

# Create the dataloaders
train_loader_size = int(len(graph_dataset)*train_ratio)
valid_loader_size = len(graph_dataset) - train_loader_size


train_loader, valid_loader = random_split(graph_dataset, [train_loader_size, valid_loader_size])

train_loader = DataLoader(dataset=train_loader, batch_size=num_in_batch, 
                          shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(dataset=valid_loader, batch_size=num_in_batch, 
                          shuffle=True, num_workers=num_workers)

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
ps = 5

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
                                                                    lr=1e-5,
                                                                    regression_label=reg_label)

train.plot_epoch_stats(epoch_stats)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define helper functions
@staticmethod
def _update_running_metrics(loss, labels, preds, running_metrics):
    running_metrics["running_loss"] += loss.item() * len(labels)
    running_metrics["running_correct"] += (preds == labels).sum().item()
    running_metrics["TP"] += (preds * labels).sum().item()
    running_metrics["TN"] += ((preds - 1) * (labels - 1)).sum().item()
    running_metrics["FP"] += (preds * (labels - 1)).sum().abs().item()
    running_metrics["FN"] += ((preds - 1) * labels).sum().abs().item()

@staticmethod
def _generate_epoch_stat(epoch, learning_rate, num_samples, running_metrics):
    TP, TN, FP, FN = (running_metrics["TP"],
                      running_metrics["TN"],
                      running_metrics["FP"],
                      running_metrics["FN"])
    
    epoch_stat = {"epoch": epoch, "lr": "{:.2E}".format(learning_rate)}
    epoch_stat["diff"] = ((TP + FP) - (TN + FN)) / num_samples
    epoch_stat["loss"] = running_metrics["running_loss"] / num_samples
    epoch_stat["accy"] = running_metrics["running_correct"] / num_samples
    epoch_stat["MCC"] = (
        np.nan
        if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0
        else (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    )
    return epoch_stat





def train_n_epochs(n_epochs, model, pred_win, train_loader, valid_loader, 
                   early_stop, early_stop_patience, lr=1e-4):

    # Define a loss function and optimizer
    if regression_label:
        criterion = nn.MSELoss()
        # model = model.float()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define a dictionary to store the dataloaders
    dataloaders_dict = {"train": train_loader, "valid": valid_loader}

    # Keep track of the best model
    best_validate_metrics = {"loss": 10.0, "accy": 0.0, "MCC": 0.0, "epoch": 0}
    best_model = copy.deepcopy(model.state_dict())
    train_metrics = {"prev_loss": 10.0, "pattern_accy": -1}
    # prev_weight_dict = {}

    # Initialize epoch data for the first run
    # Keep track of running metrics
    running_metrics = {
            "running_loss": 0.0,
            "running_correct": 0.0,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }
    
    for phase in ['train', 'valid']:
        num_samples = len(dataloaders_dict[phase].dataset)
        epoch_stat_train = _generate_epoch_stat(0, lr, num_samples, running_metrics)
        epoch_stat_valid = _generate_epoch_stat(0, lr, num_samples, running_metrics)
            

    for epoch in range(1, n_epochs+1):
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()

            # Initialize tqdm progress bar
            data_iterator = tqdm(
                dataloaders_dict[phase],
                leave=True,
                unit="batch"
            )
            data_iterator.set_description(f"Epoch {epoch}: {phase}")
            
            # Update the stats shown in the progress bar
            if phase == 'train':
                data_iterator.set_postfix({
                    "\nStats for epoch": epoch - 1,
                    "loss": epoch_stat_train["loss"],
                    "accy": epoch_stat_train["accy"],
                    "MCC": epoch_stat_train["MCC"],
                    "diff": epoch_stat_train["diff"],
                }, refresh=True)
            else:
                data_iterator.set_postfix({
                    "\nStats for epoch": epoch - 1,
                    "loss": epoch_stat_valid["loss"],
                    "accy": epoch_stat_valid["accy"],
                    "MCC": epoch_stat_valid["MCC"],
                    "diff": epoch_stat_valid["diff"],
                }, refresh=True)



            # Keep track of running metrics
            running_metrics = {
                    "running_loss": 0.0,
                    "running_correct": 0.0,
                    "TP": 0,
                    "TN": 0,
                    "FP": 0,
                    "FN": 0,
                }
            
            # for i, (data, ret5, ret20, ret60) in enumerate(data_iterator):
            for batch in data_iterator:
                # if pred_win == 5:
                #     target = ret5
                # elif pred_win == 20:
                #     target = ret20
                # elif pred_win == 60:
                #     target = ret60
                
                # images = batch['image']
                # targets = batch[f'ret{pred_win}']  # dynamic based on pred_win (5, 20, or 60)

                # Add number of color channels as 1
                # data = data.unsqueeze(1)
                # data = data.unsqueeze(1)
                
                # Transform target to [1,0] if positive movement, and [0,1] if negative movement
                # labels = (1-target).unsqueeze(1) @ torch.LongTensor([1., 0.]).unsqueeze(1).T + target.unsqueeze(1) @ torch.LongTensor([0, 1]).unsqueeze(1).T
                
                labels = batch[f'ret{pred_win}'].to(device, dtype=torch.float32)
                # labels = labels.to(torch.long)

                # # Transform target to [1,0] if positive movement, and [0,1] if negative movement
                # labels = (1 - targets).unsqueeze(1) @ torch.tensor([[1., 0.]]) + targets.unsqueeze(1) @ torch.tensor([[0., 1.]])
                
                inputs = batch['image'].to(device, dtype=torch.float32)
                # labels = labels.to(device, dtype=torch.float)
                    
                
                # labels = labels.to(device, dtype=torch.float)

                # Need the targets to be on the same device as the model later
                # target = target.to(device, dtype=torch.float)
                

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # loss = criterion(labels, outputs)
                    # loss = criterion(torch.max(labels, 1)[1], outputs)
                    
                    # # Convert one-hot encoded labels to class indices
                    # labels_indices = torch.max(labels, 1)[1]
                    
                    if not regression_label:
                        labels = torch.where(labels > 0, torch.tensor(1, device=labels.device), torch.tensor(0, device=labels.device))
                    
                    # Compute loss
                    loss = criterion(outputs, labels) 
                    
                    # preds = torch.max(outputs, 1)[1]
                    
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                if regression_label:
                    
                    preds = torch.where(outputs > 0, torch.tensor(1, device=outputs.device), torch.tensor(0, device=outputs.device)).squeeze(1)

                    labels = torch.where(labels > 0, torch.tensor(1, device=labels.device), torch.tensor(0, device=labels.device))
                
                else:
                    
                    # Convert the predictions to class labels
                    preds = torch.max(outputs, 1)[1]

                # Update the running metrics
                _update_running_metrics(loss, labels, preds, running_metrics)

                # Delete the variables to free up memory
                # del inputs, labels, data, target
            
            # Calculate the epoch statistics
            if phase == "train":
                num_samples = len(dataloaders_dict[phase].dataset)
                epoch_stat_train = _generate_epoch_stat(epoch, lr, num_samples, running_metrics)
            else:
                num_samples = len(dataloaders_dict[phase].dataset)
                epoch_stat_valid = _generate_epoch_stat(epoch, lr, num_samples, running_metrics)

            # Save the model if it has the best validation loss
            if phase == "valid":
                if epoch_stat_valid["loss"] < best_validate_metrics["loss"]:
                    for metric in ["loss", "accy", "MCC", "epoch", "diff"]:
                        best_validate_metrics[metric] = epoch_stat_valid[metric]
                    best_model = copy.deepcopy(model.state_dict())
        
        print(f'Current epoch: {epoch}. \nBest epoch: {best_validate_metrics["epoch"]}')

        if early_stop and (epoch - best_validate_metrics["epoch"]) >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break


    # Load the best model weights
    model.load_state_dict(best_model)
    best_validate_metrics["model_state_dict"] = model.state_dict().copy()

    train_metrics = evaluate(model, {"train": dataloaders_dict["train"]}, 
                             pred_win, criterion)["train"]
    train_metrics["epoch"] = best_validate_metrics["epoch"]

    del best_validate_metrics["model_state_dict"]

    return train_metrics, best_validate_metrics, model










































