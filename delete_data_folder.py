
from project_imports import *

import shutil

table_name = 'I20VolTInd20'

# Construct the directory path
dir_path = os.path.join(DATA_PATH, table_name)

# Check if the directory exists
if os.path.exists(dir_path):
    # Remove all the contents of the directory
    shutil.rmtree(dir_path)
    print(f"All data in {dir_path} has been deleted.")
else:
    print(f"The directory {dir_path} does not exist.")