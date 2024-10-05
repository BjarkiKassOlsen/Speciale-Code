
from project_imports import *

from importlib import reload
import generate_graphs
import functions

import copy

run = neptune.init_run(
    project="bjarki/Speciale",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzc0ZjBjMy0wYzBjLTQwMGYtODExYS1iNDM1MjAxZDdlNWMifQ==",
)  # your credentials

#######################################
##  Test run CNN on test set stocks  ##
#######################################

import custom_model
import custom_dataset

# Reload the modules
reload(custom_model)
reload(custom_dataset)

# Load the image data
table = 'I20VolTInd20'
market = 'US'

ws = 20

### Switch to use hdf5 file system
hdf5_train_path = f'{DATA_PATH}/{market}/{table}_train.h5'
hdf5_test_path = f'{DATA_PATH}/{market}/{table}_test.h5'

#### From previous run 

computed_mean = 0.08589867502450943
computed_std = 0.27876555919647217

####

# Define transformations
custom_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize(mean=[computed_mean], std=[computed_std]),
])

# Initialize your dataset
graph_dataset_train = custom_dataset.GraphDataset(path = hdf5_train_path, transform=custom_transforms, mode = 'train')
# graph_dataset_test = custom_dataset.GraphDataset(path = hdf5_test_path, transform=custom_transforms, mode = 'test')


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


# # DataLoader
# test_loader = DataLoader(graph_dataset_test, batch_size=num_in_batch, 
#                          shuffle=False, num_workers=num_workers, pin_memory=True)

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


n_epochs = 50
ps = 20

import train

# Reload the module
reload(train)

epoch_stats, best_validate_metrics, model = train.train_n_epochs(n_epochs = n_epochs, 
                                                                    model = CNN2D_model, 
                                                                    pred_win = ps, 
                                                                    train_loader = train_loader, 
                                                                    valid_loader = valid_loader, 
                                                                    early_stop = True, 
                                                                    early_stop_patience = 10, 
                                                                    lr=1e-4,
                                                                    regression_label=reg_label,
                                                                    run=run)

train.plot_epoch_stats(epoch_stats, run=run)

# Save the model
torch.save(model.state_dict(), f'{PROJECT_PATH}/model/LUMI_Train_NYSE_930101_200101.pth')

# Stop the monitoring on Neptune
run.stop()
