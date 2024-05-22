
from project_imports import *

from importlib import reload
import generate_graphs
import functions

import copy

run = neptune.init_run(
    project="bjarki/Speciale",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzc0ZjBjMy0wYzBjLTQwMGYtODExYS1iNDM1MjAxZDdlNWMifQ==",
)  # your credentials

# Log into the Wharton database
conn = wrds.Connection(wrds_username='bjarki')

# Setting start and end date for the data
start_date = "01/01/2022"
end_date = "12/31/2022"

reload(functions)

# Fetch data using raw_sql
sp500_daily = functions.load_sp500_data(conn, start_date, end_date, freq = 'daily')

####################################
##  Test run CNN on fewer stocks  ##
####################################

sp500_daily = sp500_daily.rename(columns={
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


# Reload the module
reload(generate_graphs)

# Define the window size to be used (input)
ws = 20

# Get out the dataset
dataset = generate_graphs.GraphDataset(df = sp500_daily, win_size=ws, mode='train', label='Ret5', 
                                       indicator = [{'MA': 20}], show_volume=True, 
                                       predict_movement=True, parallel_num=-1)


# Generate the image set
image_set, table = dataset.generate_images()

# Stop the monitoring on Neptune
run.stop()
