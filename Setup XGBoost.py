

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




######### ONLY RUN TO UPDATE EXISTING DATA #########

# Reload the module.
reload(functions)


US_market_Char = functions.load_firm_char_data(start_date, end_date, permno_unique=[])

ff5_monthly = functions.load_ff5_data(conn, start_date, end_date, freq = 'monthly')

conn.close()

#############

### Get the characteristic dataset

# Read the unzipped CSV file
csv_filename = 'signed_predictors_dl_wide.csv'
csv_file_path = os.path.join(DATA_PATH, csv_filename)
print("Reading the CSV file...")
dfChar = pd.read_csv(csv_file_path)

# ==== MERGE DATAFRAMES ====
# Assuming 'crspmsignal' dataframe already exists
print("Merging datasets...")
dfChar = pd.merge(dfChar, crspmsignal, on=['permno', 'yyyymm'], how='inner')

dfChar.to_csv(f'{DATA_PATH}/predictors_and_crspm.csv')

#################### Train and test data ####################
# Rank and scale the characteristics to have values ranging from -1 to 1, excluding 'yyyymm' and 'permno'
dfChar = dfChar.set_index(['yyyymm', 'permno']).groupby('yyyymm').rank(pct=True) * 2 - 1

# Transform the characteristics to have the cross-sectional median value if a missing value
dfChar = dfChar.fillna(df.groupby('yyyymm').transform('median'))

# Reset index to bring 'yyyymm' and 'permno' back as columns
dfChar = dfChar.reset_index()


































