
from project_imports import *

from importlib import reload
import generate_graphs
import functions

import copy

run = neptune.init_run(
    project="bjarki/Speciale",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzc0ZjBjMy0wYzBjLTQwMGYtODExYS1iNDM1MjAxZDdlNWMifQ==",
)  # your credentials

table = 'I20VolTInd20'

dataset_all = pd.read_csv(f'{DATA_PATH}/{table}_dataset.csv')

dataset_all = dataset_all.values.tolist()

generate_graphs.show_single_graph(dataset_all[5], table, run)


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

learn_data = dataset_all[dataset_all['date']<=199912]
test_data = dataset_all[dataset_all['date']>199912]

# Initialize your dataset
graph_dataset = custom_dataset.GraphDataset(learn_data, path = table, transform=custom_transforms)

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

# Start the training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ws = 20

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
                                                                    regression_label=reg_label,
                                                                    run=run)

train.plot_epoch_stats(epoch_stats, run)

# print(epoch_stats)

# Stop the monitoring on Neptune
run.stop()