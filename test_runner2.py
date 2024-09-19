
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
start_date = "01/01/2020"
end_date = "12/31/2023"

# # Query using raw_sql method
# query = f"""
#     SELECT a.*, b.date, b.prc, b.openprc, b.ret, 
#         b.askhi, b.bidlo, b.vol, b.shrout, b.cfacpr, b.cfacshr
#     FROM crsp.msp500list AS a,
#         crsp.dsf AS b
#     WHERE a.permno = b.permno
#     AND b.date >= a.start
#     AND b.date <= a.ending
#     AND b.date >= '{start_date}'
#     AND b.date <= '{end_date}'
#     ORDER BY date;
# """

# sql_engine = conn.engine
# connection = sql_engine.raw_connection()

# sp500 = pd.read_sql_query(query, connection, parse_dates=['start', 'ending', 'date'])


# db = wrds.Connection()

# Reload the module.
reload(functions)

sp500_daily = functions.load_sp500_data(conn, start_date, end_date, freq = 'daily', add_desc=False)

sp500_monthly = functions.load_sp500_data(conn, start_date, end_date, freq = 'monthly')


# market_long = conn.raw_sql(f"""
#                             SELECT b.permno, b.date, b.ret
#                             FROM crsp.msf AS b
#                             WHERE b.date >= '{start_date}'
#                             AND b.date <= '{end_date}'
#                             ORDER BY date;
#                             """, date_cols=['date'])
                            
                            



# Example DataFrame - replace this with your actual DataFrame
# sp500_daily = pd.read_csv('sp500_data.csv')  # Uncomment and modify if loading from a file

# Connect to the database
conn = pyodbc.connect(CONNECTION_STRING)
cursor = conn.cursor()

# Function to map pandas dtype to SQL
def dtype_to_sql(dtype):
    if "float" in dtype.name:
        return "FLOAT"
    elif "int" in dtype.name:
        return "INT"
    elif "datetime" in dtype.name:
        return "DATETIME"
    else:  # Default to VARCHAR for object, etc. Consider handling more types as needed.
        return "VARCHAR(255)"

create_table_query = "CREATE TABLE sp500_daily ("
for col, dtype in sp500_daily.dtypes.items():  # Using items() instead of iteritems()
    sql_dtype = dtype_to_sql(dtype)
    create_table_query += f"{col} {sql_dtype}, "
create_table_query = create_table_query.rstrip(', ') + ')'

# Display the SQL create table command
print(create_table_query)


# Cleaning float columns
float_cols = [col for col in sp500_daily.columns if 'FLOAT' in str(sp500_daily[col].dtype)]
for col in float_cols:
    # Replace infinities and NaNs
    sp500_daily[col] = sp500_daily[col].replace([np.inf, -np.inf], np.nan)
    sp500_daily[col].fillna(0, inplace=True)  # Assuming 0 is a suitable fill value; adjust as necessary.

    # Optionally, add checks for overly large numbers if needed
    # Example: sp500_daily[col] = np.clip(sp500_daily[col], -1e38, 1e38)

# Ensure all data types are correct and convert datetimes to strings if necessary
for col in sp500_daily.columns:
    if 'DATETIME' in create_table_query and col in create_table_query:
        sp500_daily[col] = sp500_daily[col].astype(str)


data_tuples = list(sp500_daily.itertuples(index=False, name=None))

# Generate placeholder for SQL insert statement
placeholders = ', '.join(['?' for _ in sp500_daily.columns])
insert_query = f"INSERT INTO sp500_daily VALUES ({placeholders})"

# Assuming 'float_cols' is correctly identified as columns with float data type
float_cols = [col for col in sp500_daily.columns if sp500_daily[col].dtype == 'float64' or sp500_daily[col].dtype == 'float32']
for col in float_cols:
    sp500_daily[col] = sp500_daily[col].apply(lambda x: None if pd.isna(x) else float(x))

# Ensure data is correctly converted by checking some rows
print(sp500_daily[float_cols].head())

data_tuples = [tuple(x) for x in sp500_daily.to_numpy()]

placeholders = ', '.join(['?' for _ in sp500_daily.columns])
insert_query = f"INSERT INTO sp500_daily VALUES ({placeholders})"

# Attempt to upload data, logging each tuple
try:
    for data_tuple in data_tuples:
        # Log the data tuple right before executing
        print("Inserting data:", data_tuple)
        cursor.execute(insert_query, data_tuple)
    conn.commit()
    print("Data uploaded successfully.")
except Exception as e:
    print(f"An error occurred while inserting data: {e}")
    print(f"Problematic data: {data_tuple}")
    conn.rollback()  # Ensures that partial data isn't left in the database


# Close the connection
cursor.close()
conn.close()



# table_name = 'I20VolTInd20'

# conn = pyodbc.connect(CONNECTION_STRING)
# cursor = conn.cursor()
# cursor.execute(f"DROP TABLE {table_name}")
# conn.commit()  # Commit the transaction


# market_long2 = conn.raw_sql(f"""
#                             SELECT a.*, b.date, b.ret
#                             FROM crsp.msp500list AS a,
#                             crsp.msf AS b
#                             WHERE a.permno = b.permno
#                             AND b.date >= '{start_date}'
#                             AND b.date <= '{end_date}'
#                             ORDER BY date;
#                             """, date_cols=['start', 'ending', 'date'])

# Get a list of all companies
permno_list = sp500.permno.unique()

# Get a list of all companies in CRSP on exhanges 1, 2, 3
permno_list_long = market_long.permno.unique()


# Define the directory path
directory_path = os.path.join(DATA_PATH, table_name)

# Iterate over all files in the directory and delete only .png files
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path) and filename.endswith('.png'):
        os.remove(file_path)


# a = permno_list_long[:10000]

# a = np.unique(np.concatenate((permno_list_long, permno_list)))

# Load the firm characteristics
# dfChar = functions.load_firm_char_data()
# dfChar = functions.load_firm_char_data(permno_list)
# dfChar = functions.load_firm_char_data(199212, 202301, a)
# dfChar = pd.read_csv(f'{DATA_PATH}/firm_characteristics/combined_df.csv')

zip_path = f'{DATA_PATH}/signed_predictors_dl_wide.zip'
csv_file = 'signed_predictors_dl_wide.csv'

# Create a temporary directory to extract the file
with ZipFile(zip_path, 'r') as zip_ref:
    # Extract all the contents into the directory
    zip_ref.extractall(DATA_PATH)

# Read the extracted CSV file
dfChar = pd.read_csv(os.path.join(DATA_PATH, csv_file))

# # Remove the extracted file, for space on harddrive
# os.remove(os.path.join(DATA_PATH, csv_file))


# Reload the module
reload(functions)

ff5 = functions.load_ff5_data(conn, start_date, end_date)


test = sp500[sp500['permno'].isin([79764, 85913])]


grouped = sp500.groupby('comnam')['permno'].unique()

# Filter to find companies with more than one 'permno' associated
grouped[grouped.apply(len) > 1]





########################################
##  Test run XGBoost on fewer stocks  ##
########################################

######
# Preprocess of data
######


# See how many characteristics have an empty value
pd.set_option('display.max_rows',None)
dfChar.isna().sum().sort_values(ascending = True)

# Rank and scale the characteristics to have values ranging from -1 to 1, excluding 'yyyymm' and 'permno'
df = dfChar.set_index(['yyyymm', 'permno']).groupby('yyyymm').rank(pct=True) * 2 - 1

# Transform the characteristics to have the cross-sectional median value if a missing value
df = df.fillna(df.groupby('yyyymm').transform('median'))

# Reset index to bring 'yyyymm' and 'permno' back as columns
df = df.reset_index()

# Only keep sp500 companies
df = df[df.permno.isin(permno_list)]

# See how many characteristics have an empty value
df.isna().sum().sort_values(ascending = True)

# Start from 1993 as the CNN data is first all available there
dfChar = df[df['yyyymm']>199612]
dfChar = dfChar[dfChar['yyyymm']<202101]



# Drop the variables with more than 25% missing values
# limitPer = len(df) * .9
# dfChar = dfChar.dropna(axis=1)

# See how many characteristics have an empty value
dfChar.isna().sum().sort_values(ascending = True)

len(df.permno.unique())

########
# Start training process
########

# # Add the returns onto the companies
# df['ret'] = df[df.groupby('permno') == sp500.permno]


# Utilize the returns from the market_long dataset, as we can retain more information then
sp500_long = market_long[market_long.permno.isin(sp500.permno.unique())]

# Shift the returns one month back
sp500_long['ret_1m_ahead'] = sp500_long.groupby('permno')['ret'].shift(-1)


# Merge the 'ret_1m_ahead' from 'sp500_long' into 'sp500' based on 'yyyymm' and 'permno'
sp500 = pd.merge(sp500, sp500_long[['date', 'permno', 'ret_1m_ahead']], on=['date', 'permno'], how='left')


# Convert date to same type
sp500['date'] = pd.to_datetime(sp500['date'], format='%Y-%m-%d-%H-%M-%S').dt.to_period('M')
sp500['start'] = pd.to_datetime(sp500['start'], format='%Y-%m-%d-%H-%M-%S').dt.to_period('M')
sp500['ending'] = pd.to_datetime(sp500['ending'], format='%Y-%m-%d-%H-%M-%S').dt.to_period('M')

# Converting 'yyyymm' to 'YYYY-MM'
dfChar['date'] = pd.to_datetime(dfChar['yyyymm'], format='%Y%m').dt.to_period('M')

# Perform the merge to include only the 'ret' column from 'sp500' into 'df'
merged_df = pd.merge(dfChar, sp500[['date', 'permno', 'ret_1m_ahead', 'start', 'ending']], 
                     on=['date', 'permno'], how='left')

# # Shift the returns one month back
# merged_df['ret_1m_ahead'] = merged_df.groupby('permno')['ret'].shift(-1)

# Filter to keep rows where 'yyyymm' falls within the 'start_yyyymm' and 'ending_yyyymm' period
filtered_df = merged_df[(merged_df['date'] >= merged_df['start']) & (merged_df['date'] < merged_df['ending'])]

# See how many characteristics have an empty value
filtered_df.isna().sum().sort_values(ascending = True)

# Drop rows that don't have the pedictive value available
cleaned_df = merged_df.dropna(subset=['ret_1m_ahead'])

# Filter rows where any column has NaN values
# a = filtered_df[filtered_df.isna().any(axis=1)]

# a = sp500[sp500.ret.isna() == True]

# b = filtered_df[filtered_df.ret.isna() == True]

# c = filtered_df[filtered_df.ret_1m_ahead.isna() == True]

# d = filtered_df[(filtered_df['ret'].isna()) | (filtered_df['ret_1m_ahead'].isna())]


########
# Initialize training and valid dataset
########

# Choose some cutoff period
dfTrain = cleaned_df[cleaned_df.date < '2000-01']
dfTest = cleaned_df[cleaned_df.date == '2000-01']

# Define the columns to exclude from the features
exclude_columns = ['yyyymm', 'date', 'permno', 'start', 'ending']

# Separate features and target for the training data
X = dfTrain.drop(columns=exclude_columns + ['ret_1m_ahead'])
y = dfTrain['ret_1m_ahead']

# Split the training data into training and validation sets
train_ratio = 0.8
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=train_ratio, random_state=42)



import xgboost as xgb

# Convert the pandas DataFrames into DMatrix objects; XGBoost's optimized data structure
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# Specify your training parameters
params = {
    'max_depth': 10,
    'eta': 0.05,
    'objective': 'reg:squarederror',  # Use 'reg:linear' for older versions of XGBoost
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda'
}

# Train the model
evallist = [(dvalid, 'eval'), (dtrain, 'train')]
num_round = 1000
bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=100)

# Make predictions

# Prepare DMatrix for the test set
dtest = xgb.DMatrix(dfTest.drop(columns=exclude_columns + ['ret_1m_ahead']))

# Make predictions
dfTest['predicted_ret'] = bst.predict(dtest)

# Function to form portfolios based on predictions
def form_portfolios(df):
    df_sorted = df.sort_values(by='predicted_ret', ascending=False)
    high = df_sorted.head(int(len(df_sorted) * 0.1))  # top 10%
    low = df_sorted.tail(int(len(df_sorted) * 0.1))  # bottom 10%
    
    return pd.Series({
        'Top Portfolio': high['ret_1m_ahead'].mean(),
        'Bottom Portfolio': low['ret_1m_ahead'].mean(),
        'Long-Short': high['ret_1m_ahead'].mean() - low['ret_1m_ahead'].mean()
    })

# Applying the portfolio formation function to each month in the test data
portfolio_returns = dfTest.groupby(dfTest['date']).apply(form_portfolios).reset_index()

# Display the DataFrame
print(portfolio_returns['Long-Short'].mean())




#################################
## Cross Validation for params ##
#################################

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Define parameters for the grid search
param_grid = {
    'max_depth': [3, 5, 7, 10],  # Depth of each tree
    'eta': [0.01, 0.1, 0.3],     # Learning rate
    'subsample': [0.5, 0.7, 1.0],# Subsample percentage for each tree
    'colsample_bytree': [0.5, 0.7, 1.0],  # Subsample ratio of columns when constructing each tree
}

# Setting up the scenario for GridSearchCV
# Assuming dfTrain is defined as before with the proper preprocessing steps
X_train = dfTrain.drop(columns=exclude_columns + ['ret_1m_ahead', 'ret_1m_ahead_binary'])
y_train = (dfTrain['ret_1m_ahead'] > 0).astype(int)

# Convert data to DMatrix form
dtrain = xgb.DMatrix(X_train, label=y_train)

# Instantiate an XGBoost classifier object
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Set up GridSearchCV
clf = GridSearchCV(xgb_model, param_grid, n_jobs=-1, cv=3, scoring='roc_auc')

# Perform grid search
clf.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters:", clf.best_params_)
print("Best score:", clf.best_score_)

# Use best parameters to train the final model
final_model = xgb.XGBClassifier(**clf.best_params_, objective='binary:logistic', eval_metric='logloss')
final_model.fit(X_train, y_train)

# Optionally: Predict on test data or validation data
# X_test = dfTest.drop(columns=exclude_columns + ['ret_1m_ahead'])
# y_test = (dfTest['ret_1m_ahead'] > 0).astype(int)
# predictions = final_model.predict(X_test)

##############






# Parameters
train_ratio = 0.8
params = {
    'max_depth': 10,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'gpu'
}
# num_round = 500
portfolio_returns_list = []

# Define the columns to exclude from the features
exclude_columns = ['yyyymm', 'date', 'permno', 'start', 'ending']

# Rolling training and prediction
unique_years = sorted(cleaned_df['date'].dt.year.unique())
for i in range(len(unique_years) - 1):
    training_end_year = unique_years[i]
    prediction_year = unique_years[i + 1]
    
    print(f'Prediction year: {prediction_year}')
    
    # Increasing num_round every year
    num_round = 2000 + (500 * i)

    # Splitting the data
    dfTrain = cleaned_df[cleaned_df['date'].dt.year <= training_end_year]
    dfTest = cleaned_df[cleaned_df['date'].dt.year == prediction_year]
    
    # Separate features and target for the training data
    X_train = dfTrain.drop(columns=exclude_columns + ['ret_1m_ahead', 'ret_1m_ahead_binary'])
    y_train = dfTrain['ret_1m_ahead']

    # Validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=train_ratio, random_state=42)
    
    # Prepare DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Training the model
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=30)

    # Preparing test set and making predictions
    X_test = dfTest.drop(columns=exclude_columns + ['ret_1m_ahead', 'ret_1m_ahead_binary'])
    dtest = xgb.DMatrix(X_test)
    dfTest['predicted_ret'] = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

    # Function to form portfolios based on predictions
    def form_portfolios(df):
        df_sorted = df.sort_values(by='predicted_ret', ascending=False)
        high = df_sorted.head(int(len(df_sorted) * 0.1))
        low = df_sorted.tail(int(len(df_sorted) * 0.1))
        return pd.Series({
            'Top Portfolio': high['ret_1m_ahead'].mean(),
            'Bottom Portfolio': low['ret_1m_ahead'].mean(),
            'Long-Short': high['ret_1m_ahead'].mean() - low['ret_1m_ahead'].mean()
        })

    # Applying the portfolio formation function to each month in the test data
    monthly_returns = dfTest.groupby(dfTest['date']).apply(form_portfolios)
    monthly_returns['Year'] = prediction_year
    portfolio_returns_list.append(monthly_returns)

# Combine all years' returns
portfolio_returns = pd.concat(portfolio_returns_list)

# Reset index and rename columns for clarity
portfolio_returns.reset_index(inplace=True)

# Display the DataFrame
print(portfolio_returns['Long-Short'].mean())


# Assuming monthly returns can simply be summed up to simulate compounding
portfolio_returns['Cumulative Return'] = (1 + portfolio_returns['Long-Short']).cumprod() * 100

# Plotting the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(portfolio_returns['date'].dt.to_timestamp(), portfolio_returns['Cumulative Return'], marker='o', linestyle='-')
plt.title('Cumulative Returns of Long-Short Portfolio Strategy')
plt.xlabel('Month')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate dates for better visibility
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
plt.show()



####################################
##  Test XGB for classification   ##
####################################

# Parameters for XGBoost
# params = {
#     'max_depth': 10,
#     'eta': 0.1,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'objective': 'binary:logistic',
#     'eval_metric': 'logloss',
#     'gamma': 10, 
#     'lambda': 1.0,
#     'alpha': 0.1,
#     'tree_method': 'hist',
#     'device': 'cuda'
# }

params = {
    'max_depth': 10,
    'eta': 0.1,
    'objective': 'binary:logistic',  # Updated for classification
    'eval_metric': 'logloss',        # Updated for classification
    'tree_method': 'hist',
    'device': 'cuda'
}
portfolio_returns_list = []

# Define the columns to exclude from the features
exclude_columns = ['yyyymm', 'date', 'permno', 'start', 'ending']

# Convert returns to binary outcomes
cleaned_df['ret_1m_ahead_binary'] = (cleaned_df['ret_1m_ahead'] > 0).astype(int)

# Rolling training and prediction
unique_years = sorted(cleaned_df['date'].dt.year.unique())
for i in range(len(unique_years) - 1):
    training_end_year = unique_years[i]
    prediction_year = unique_years[i + 1]
    
    print(f'Prediction year: {prediction_year}')
    
    # Increasing num_round every year
    num_round = 2000 + (500 * i)

    # Splitting the data
    dfTrain = cleaned_df[cleaned_df['date'].dt.year <= training_end_year].copy()
    dfTest = cleaned_df[cleaned_df['date'].dt.year == prediction_year].copy()
    
    # Separate features and target for the training data
    X_train = dfTrain.drop(columns=exclude_columns + ['ret_1m_ahead', 'ret_1m_ahead_binary'])
    y_train = dfTrain['ret_1m_ahead_binary']

    # Validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=train_ratio, random_state=42)
    
    # Prepare DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Training the model
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=30)

    # Preparing test set and making predictions
    X_test = dfTest.drop(columns=exclude_columns + ['ret_1m_ahead', 'ret_1m_ahead_binary'])
    dtest = xgb.DMatrix(X_test)
    
    # Ensure to use the best iteration
    dfTest['predicted_ret'] = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

    # Function to form portfolios based on predictions
    def form_portfolios(df):
        df_sorted = df.sort_values(by='predicted_ret', ascending=False)
        high = df_sorted.head(int(len(df_sorted) * 0.1))
        low = df_sorted.tail(int(len(df_sorted) * 0.1))
        return pd.Series({
            'Top Portfolio': high['ret_1m_ahead'].mean(),
            'Bottom Portfolio': low['ret_1m_ahead'].mean(),
            'Long-Short': high['ret_1m_ahead'].mean() - low['ret_1m_ahead'].mean()
        })

    # Applying the portfolio formation function to each month in the test data
    monthly_returns = dfTest.groupby(dfTest['date']).apply(form_portfolios)
    monthly_returns['Year'] = prediction_year
    portfolio_returns_list.append(monthly_returns)

# Combine all years' returns
portfolio_returns = pd.concat(portfolio_returns_list)

# Reset index and rename columns for clarity
portfolio_returns.reset_index(inplace=True)

# Display the DataFrame
print("Average Long-Short Portfolio Return:", portfolio_returns['Long-Short'].mean())

# Assuming monthly returns can simply be summed up to simulate compounding
portfolio_returns['Cumulative Return'] = (1 + portfolio_returns['Long-Short']).cumprod() * 100

# Plotting the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(portfolio_returns['date'].dt.to_timestamp(), portfolio_returns['Cumulative Return'], marker='o', linestyle='-')
plt.title('Cumulative Returns of Long-Short Portfolio Strategy')
plt.xlabel('Month')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate dates for better visibility
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
plt.show()





####################################
##  Test run CNN on fewer stocks  ##
####################################


# df = sp500[sp500['permno'].isin(permno_list)].copy()


# dfApplMSFT.reset_index(drop=True)


# Only keep the columns we need
# df = dfApplMSFT[['date', 'prc', 'openprc', 'askhi', 'bidlo', 'vol', 'ret5', 'ret20', 'ret60']].copy()
# df = dfApplMSFT.copy()
# df = sp500.copy()

# Keep the permno
# permno = df['permno'].unique()

# Rename columns
# df['date', 'prc', 'openprc', 'askhi', 'bidlo', 'vol', 'ret5', 'ret20', 'ret60'] = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Ret5', 'Ret20', 'Ret60']

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

first_trading_day = "31-05-2022"

# Get out the dataset
dataset = generate_graphs.GraphDataset(df = sp500_daily, win_size=ws, mode='train', label='Ret20', 
                                       indicator = [{'MA': 20}], show_volume=True, 
                                       predict_movement=True, cut_off_date=first_trading_day, parallel_num=-1)


# Generate the image set
image_set, table = dataset.generate_images()

# # Extract the dataset
# dataset_all = image_set[0]

# # Extract the save_path
# table = image_set[1]


# save_path = f'{DATA_PATH}/I20VolTInd20'

dataset_all = pd.read_csv(f'{DATA_PATH}/{table}_dataset.csv')

# Extract date and convert to datetime
dataset_all['date'] = pd.to_datetime(dataset_all.iloc[:,0].str.split('_').str[-1].str.replace('.png', ''), format='%Y-%m-%d')
# dataset_all['date'] = pd.to_datetime(dataset_all.iloc[:,0].str.extract(r'(\d{4}-\d{2})')[0], format='%Y-%m')


# Split the dataset
learn_data = dataset_all[dataset_all['date'] <= '2022-05']
test_data = dataset_all[dataset_all['date'] > '2022-05']


learn_data = learn_data.values.tolist()
test_data = test_data.values.tolist()

# # Plot a graph
# generate_graphs.show_single_graph(image_set[0][0])


# # Concatenate the dataset
# dataset_all = []
# for data in dataset_all2:
#     print(data)
#     dataset_all.append(data)
# dataset_all2 = []



# for graph in dataset_all:
#     generate_graphs.show_single_graph(graph)

generate_graphs.show_single_graph(learn_data[5], table)

# data = h5py.File(h5_filename)

# np.array(data[table]['FileName'])[4]
# np.array(data[table]['FileData'])[4]


import pyodbc

conn = pyodbc.connect(CONNECTION_STRING)
cursor = conn.cursor()

# cursor.execute(f"CREATE TABLE IF NOT EXISTS I20VolTInd20 (ID VARCHAR(50) NOT NULL PRIMARY KEY, Image VARBINARY(MAX) NOT NULL)")

try:
    # Assuming `cursor` is a cursor object connected to a SQL Server database
    sql_command = """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'I20VolTInd20' AND type = 'U')
    BEGIN
        CREATE TABLE I20VolTInd20 (
            ID VARCHAR(50) NOT NULL PRIMARY KEY,
            Image VARBINARY(MAX) NOT NULL
        )
    END
    """
    cursor.execute(sql_command)
    cursor.commit()
except Exception as e:
    print("An error occurred:", e)
finally:
    cursor.close()

image_path = f'{DATA_PATH}/I20VolTInd20/'

# Fetch all PNG filenames
filenames = [f for f in os.listdir(image_path) if f.endswith('.png')]

# Iterate over the files in the specified directory with a progress bar
for filename in tqdm(filenames, desc="Uploading images"):
    if filename.endswith('.png'):  # Check if the file is a PNG image
        file_id = filename.split('.')[0]  # Assuming the ID is the part of the filename before '.png'
        full_path = os.path.join(image_path, filename)

        with open(full_path, 'rb') as file:
            binary_data = file.read()

        try:
            # SQL command to insert data
            sql_insert_blob_query = f"""
            INSERT INTO {table} (ID, Image) VALUES (?, ?)
            """
            # Executing the SQL command and committing changes
            cursor.execute(sql_insert_blob_query, (file_id, binary_data))
            cursor.commit()
            # print(f"Successfully uploaded {filename}")
        
        except Exception as e:
            print(f"Failed to upload {filename}: {e}")

cursor.close()
conn.close()
print("Database connection closed.")


# def inspect_h5_file(h5_filename):
#     with h5py.File(h5_filename, 'r') as h5_file:
#         def print_structure(name, obj):
#             print(name)
#         h5_file.visititems(print_structure)
        
# inspect_h5_file(h5_filename)

#############################
##  Save the data locally  ##
#############################


# # Assuming your dataframe and path are defined as follows:
# save_path = f'{DATA_PATH}/I20VolTInd20/'
# # dataframe_path = 'path_to_your_dataframe.csv'  # Modify this path accordingly

# # # Load the dataframe
# # dataset_all = pd.read_csv(dataframe_path)

# # Define the HDF5 file path
# hdf5_path = f'{DATA_PATH}/{table}_images.h5'

# # Open an HDF5 file
# with h5py.File(hdf5_path, 'w') as h5f:
#     # Create datasets for images and labels
#     # Assuming images are 256x256 and grayscale
#     images_dset = h5f.create_dataset('images', (len(dataset_all), 64, 60), dtype='uint8', compression='gzip')
#     labels_dset = h5f.create_dataset('labels', (len(dataset_all), 3), dtype='float32', compression='gzip')

#     # Iterate through each row in the DataFrame, load the image, and save it along with the labels
#     for i, row in tqdm(dataset_all.iterrows(), total=len(dataset_all), desc='Processing images'):
#         # Load image
#         img_path = os.path.join(save_path, row[0])
#         with Image.open(img_path) as img:
#             img_array = np.array(img)

#         # # Check if the image needs resizing
#         # if img_array.shape[0] != 64 or img_array.shape[1] != 60:
#         #     img = img.resize((64, 60))
#         #     img_array = np.array(img)

#         # Store the image in the dataset
#         images_dset[i, :, :] = img_array

#         # Store the labels in the dataset
#         labels_dset[i] = row[1:4]  # Assuming the labels are in columns 1, 2, and 3

# print("All data has been saved to HDF5.")




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

# learn_data = dataset_all[dataset_all['date']<=202204]
# test_data = dataset_all[dataset_all['date']>202204]

# Define transformations
custom_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
])

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


test_data














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










































