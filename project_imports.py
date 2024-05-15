
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import os

from zipfile import ZipFile
from tempfile import TemporaryDirectory

import wrds

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from sklearn.model_selection import train_test_split

from PIL import Image

from tqdm import tqdm
from joblib import Parallel, delayed

import h5py

import io
import pyodbc
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool


###############################
## Define the relevant paths ##
###############################

def get_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir

# PROJECT_PATH = get_dir('C:/Users/bjark/Documents/AU/Kandidat/4. Semester/Code/Speciale-Code')
PROJECT_PATH = get_dir('/scratch/project_465001092')
DATA_PATH = get_dir(os.path.join(PROJECT_PATH, "data"))



###########################
## Conection to Database ##
###########################

# Define the connection string
# server = 'speciale-data.database.windows.net'
# database = 'Data'
# username = 'JB5247'
# password = 'ExMWQtVtA7NTQ9k'
driver= '{ODBC Driver 18 for SQL Server}'

# CONNECTION_STRING = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
CONNECTION_STRING = f'DRIVER={driver};SERVER=speciale-data.database.windows.net;DATABASE=Data;UID=JB5247;PWD=ExMWQtVtA7NTQ9k'
POOL_CONNECTION_STRING = "mssql+pyodbc://JB5247:ExMWQtVtA7NTQ9k@speciale-data.database.windows.net/Data?driver=ODBC+Driver+18+for+SQL+Server"
