# -*- coding: utf-8 -*-

import sys
sys.path.append('C:/Users/bjark/Documents/AU/Kandidat/4. Semester/Code/Speciale-Code')

from project_imports import *

from importlib import reload
import generate_graphs
import functions


def update():
    
    # Log into the Wharton database
    conn = wrds.Connection(wrds_username='bjarki')
    
    # Reload the required modules
    reload(functions)
    reload(generate_graphs)
    
    
    ############ Load crspm returns
    
    # Setting start and end date for the data
    start_date = "01/01/1992"
    end_date = "01/01/2023"
    
    # Get Fama French monthly data
    ff5m = functions.load_ff5_data(conn, start_date, end_date, freq = 'monthly')
    
    # Get US market data, adjusted for delisting
    crspm = functions.load_returns_crspm(conn, start_date, end_date, rf = ff5m[['yyyymm', 'rf']]) # 1.829.225 rows
    
    
    ############ Generate image data
    
    # Set initial parameters
    start_year = 1992
    initial_end_year = 2023
    window_size = 4
    overlap = 1
    drop_rate = 0.0
    adj_prc = False
    
    # Reload the required modules
    reload(functions)
    reload(generate_graphs)
    
    dates_to_predict = list(crspm.date.unique())
    
    # Initialize an empty list to collect the rows for the DataFrame
    data_rows = []
    
    # Use a set to track unique (permno, date) combinations
    unique_combinations = set()
    
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
        
        US_market = functions.load_US_market(conn, start_date, end_date, freq = 'daily', add_desc=False, ret = True)
        
        US_market = US_market.rename(columns={
            'date': 'Date', 
            'prc': 'Close', 
            'openprc': 'Open', 
            'askhi': 'High', 
            'bidlo': 'Low', 
            'vol': 'Volume'
        })
        
        US_market.Close = US_market.Close.abs()
        
        # Reload the module
        reload(generate_graphs)
        
        # Define the window size to be used (input)
        ws = 20
        
        market = 'US'
        
        # Get out the dataset
        dataset_gen = generate_graphs.GraphDataset(df = US_market, win_size=ws, mode='train', label='Ret20', market = market,
                                               indicator = [{'MA': 20}], show_volume=True, drop_rate = drop_rate, adj_prc=adj_prc,
                                               predict_movement=True, dates_to_gen=dates_to_predict, parallel_num=-1)
        
        # Generate the image set
        dataset_append = dataset_gen.generate_images()
        
        # Iterate through the generated dataset_append and populate the DataFrame structure with a progress bar
        for data in tqdm(dataset_append, desc="Processing generated datasets", total=len(dataset_append)):
            if len(data) > 0:
                for image_array, permno, date_str in data:
                    # Check if the (permno, date) combination already exists
                    if (permno, date_str) not in unique_combinations:
                        # Add the combination to the set
                        unique_combinations.add((permno, date_str))
                        
                        # Append a new row as a dictionary to `data_rows`
                        data_rows.append({'permno': permno, 'date': date_str, 'image': image_array})
    
        
        # Move to the next start year (this creates the 1-year overlap)
        start_year += window_size - overlap - 1
    
    
    # Create a DataFrame from the list of dictionaries
    dataset = pd.DataFrame(data_rows)
    
    # Close database connection
    conn.close()
    
    # Clean up memory
    US_market = []
    
    
    
    
    
    ########## Filter down the data to the wanted analysis dates
    
    # Setting start and end date for the data
    start_date = "01/01/1993"
    end_date = "01/01/2022"
    
    
    
    # Convert byte strings to strings and then to datetime format
    dataset['date'] = pd.to_datetime(dataset['date'].str.decode('utf-8'))
    # Decode byte strings to UTF-8 strings and then convert to integers
    dataset['permno'] = dataset['permno'].str.decode('utf-8').astype('int64')
    
    
    # Filter the DataFrame for dates within the specified range
    dataset = dataset[(dataset['date'] >= pd.to_datetime(start_date)) & (dataset['date'] <= pd.to_datetime(end_date))] # 1.703.454 rows
    
    # Filter the crspm dataset
    crspm = crspm[(crspm['date'] >= pd.to_datetime(start_date)) & (crspm['date'] <= pd.to_datetime(end_date))] # 1.711.454 rows
    
    # =============== Analysing differences in datasets ============
    
    # # Calculate matching pairs
    # common_pairs = set(zip(crspm_filtered['permno'], crspm_filtered['date'])) & set(zip(dataset['permno'], dataset['date']))
    # print("Common pairs count:", len(common_pairs))
    
    # # Create sets of (permno, date) tuples from each DataFrame
    # crspm_pairs = set(zip(crspm_filtered['permno'], crspm_filtered['date']))
    # dataset_pairs = set(zip(dataset['permno'], dataset['date']))
    
    # # Find unique pairs in each DataFrame
    # unique_to_crspm = crspm_pairs - dataset_pairs
    # unique_to_dataset = dataset_pairs - crspm_pairs
    
    # # Combine unique sets to get all unique pairs across both DataFrames
    # not_common_pairs = unique_to_crspm.union(unique_to_dataset)
    
    # ===============================================================
    
    
    # Filter datasets to keep only rows with matching (permno, date) pairs
    crspm = crspm[crspm[['permno', 'date']].apply(tuple, axis=1).isin(set(zip(dataset['permno'], dataset['date'])))] # 1.690.483 rows
    
    dataset = dataset[dataset[['permno', 'date']].apply(tuple, axis=1).isin(set(zip(crspm['permno'], crspm['date'])))] # 1.690.483 rows
    
    ### Get the characteristic dataset
    
    # Read the unzipped CSV file
    csv_filename = 'signed_predictors_dl_wide.csv'
    csv_file_path = os.path.join(DATA_PATH, csv_filename)
    print("Reading the CSV file...")
    dfChar = pd.read_csv(csv_file_path)
    
    # ==== MERGE DATAFRAMES ====
    # Assuming 'crspmsignal' dataframe already exists
    print("Merging datasets...")
    # dfChar = pd.merge(dfChar, crspmsignal, on=['permno', 'yyyymm'], how='inner')
    
    # # Create a set of (permno, yyyymm) pairs that exist in `crspm`
    # matching_pairs = set(zip(crspm['permno'], crspm['yyyymm']))
    
    # Filter `dfChar` to keep only rows with matching (permno, yyyymm) pairs
    dfChar = dfChar[dfChar[['permno', 'yyyymm']].apply(tuple, axis=1).isin(set(zip(crspm['permno'], crspm['yyyymm'])))] # 1.690.483 rows
    
    
    dfChar.to_csv(f'{DATA_PATH}/predictors_and_crspm.csv')
    
    #########
    
    # dfChar.read_csv(f'{DATA_PATH}/predictors_and_crspm.csv')
    
    # Rank and scale the characteristics to have values ranging from -1 to 1, excluding 'yyyymm' and 'permno'
    dfChar = dfChar.set_index(['yyyymm', 'permno']).groupby('yyyymm').rank(pct=True) * 2 - 1
    
    # Transform the characteristics to have the cross-sectional median value if a missing value
    dfChar = dfChar.fillna(dfChar.groupby('yyyymm').transform('median'))
    
    dfChar.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    dfChar = dfChar.fillna(dfChar.groupby('yyyymm').transform('median'))
    
    # Reset index to bring 'yyyymm' and 'permno' back as columns
    dfChar = dfChar.reset_index()
    
    
    # ========= Combine data
    
    dataset['yyyymm'] = (dataset['date'].dt.year * 100 + dataset['date'].dt.month).astype('int64')
    
    # Cut down on columns and merge
    merged_df = pd.merge(dataset[['permno', 'yyyymm', 'image']], crspm[['permno', 'yyyymm', 'excess_ret_ahead', 'me']], on=['permno', 'yyyymm'], how='inner')
    
    # Convert the data to uint8 before flattening
    dfChar['chars'] = dfChar.iloc[:, 2:].astype('float32').apply(lambda x: list(x), axis=1)
    
    # Now merge only the flattened column
    merged_df = pd.merge(dfChar[['permno', 'yyyymm', 'chars']], merged_df, on=['permno', 'yyyymm'], how='inner') # 1.690.483 rows
    
    
    ##### Add everything to a single hdf5 file
    
    # Specify the data path
    table = 'I20VolTInd20'
    
    # Setup the path to the dataset
    hdf5_dataset_path = f'{DATA_PATH}/{market}/{table}_dataset.h5'
    
    
    # Extract the relevant data
    images = merged_df['image'].tolist()
    chars = merged_df['chars'].tolist()  # replace 'chars_column_name' with actual column name
    excess_ret_ahead = merged_df['excess_ret_ahead'].values.astype(np.float32)
    permnos = merged_df['permno'].astype(str).values
    dates = merged_df['yyyymm'].astype(str).values
    me = merged_df['me'].values.astype(np.float32)
    
    # Create and write to HDF5 file
    with h5py.File(hdf5_dataset_path, 'w') as h5_file:
        h5_file.create_dataset('chars', data=chars, maxshape=(None,209), dtype=np.float32)
        h5_file.create_dataset('images', data=images, maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
        h5_file.create_dataset('labels', data=excess_ret_ahead, maxshape=(None,), dtype=np.float32)
        h5_file.create_dataset('permnos', data=permnos, maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
        h5_file.create_dataset('dates', data=dates, maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
        h5_file.create_dataset('ME', data=me, maxshape=(None,), dtype=np.float32)
    
    print("Data successfully saved to HDF5.")
    
        
    assert len(images) == len(chars) == len(excess_ret_ahead) == len(permnos) == len(dates) == len(me), "Data arrays must all be the same length"