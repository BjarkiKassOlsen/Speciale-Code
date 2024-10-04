from project_imports import *


# Description: This file contains the functions to generate the graphs for each stock



def image_generator(permno, df, image_size, lookback, indicator, show_volume, pred_move, 
                    save_path, train_file_path, test_file_path, mode, cut_off_date):
    """
    Generate the graphs for each stock

    input:
        df: dataframe with the stock data
        image_size: size of the image to output
        start_date: start date of the image to generate
        end_date: end date of the image to generate
        indicators: list of indicators to plot, i.e. ['MA': 20]
        show_volume: boolean to show the volume in the graph
        mode: 'train' or 'test'

    output:

    """    
    
    if len(indicator) > 0:
        for ind in indicator:
            for key, value in ind.items():
                df = df.assign(**{key: df['Close'].rolling(window=value).mean()})
                
                # Drop the starting NA's as they will no longer have any data on MA
                df = df[value-1:]
             
    # Skip the generation if the DataFrame has fewer than 40 entries
    if df.shape[0] < lookback:
        return  # Exit the function early
             

    # Remove rows with NA's
    # df = df.dropna()
    
    # Convert to date
    cut_off_date_dt = pd.to_datetime(cut_off_date, format='%d-%m-%Y')
    
    # # Split data to train and test data
    # df_train = df[df['Date'] <= cut_off_date_dt]
    # df_test = df[df['Date'] > cut_off_date_dt]

    # dataset = []
    
    # # Prepare the test dataset for each last trading day
    # test_dataset = []
    
    # Group by year and month to find the last trading day
    last_trading_days = df.groupby([df['Date'].dt.year, df['Date'].dt.month])['Date'].max()
    
    last_trading_days = last_trading_days[last_trading_days>=cut_off_date_dt]
    
    # Create dataset if not already existing
    train_len = 2500 # Number of buisness days between 1993-01-01 to 2001-01-01
    test_len = 230 # Number of months between same range
    
    permno_str = str(permno)
    
    dates_train = []
    dates_test = []
    
    # Open both HDF5 files, one for training data and one for testing data
    with h5py.File(os. train_file_path, 'a') as train_file, h5py.File(test_file_path, 'a') as test_file:
    
        if df.Date.iloc[lookback] < cut_off_date_dt: # Check if we have training data
            # Create a new group for the permno if it doesn't exist
            if permno_str not in train_file:
                group_train = train_file.create_group(permno_str)
            else:
                group_train = train_file[permno_str]    
            
            # Create datasets for images, labels, and dates inside the permno group
            if "images" not in group_train:
                group_train.create_dataset("images", (train_len,) + image_size, dtype=np.uint8)
                group_train.create_dataset("labels", (train_len,), dtype=np.float32)
                hdf5_file.create_dataset("permnos", (total_len,), dtype=h5py.string_dtype(encoding='utf-8'))
                group_train.create_dataset("dates", (train_len,), dtype=h5py.string_dtype(encoding='utf-8'))
                
            dates_train = list(group_train["dates"][:])
            
            # Get the last used index, for attachment
            last_used_index_train = len([date for date in dates_train if date != b''])
        
        
        if len(last_trading_days) > 2: # Check if we have test data
        
            if permno_str not in test_file:
                group_test = test_file.create_group(permno_str)
            else:
                group_test = test_file[permno_str]
                
            # Create datasets for images, labels, and dates inside the permno group
            if "images" not in group_test:
                group_test.create_dataset("images", (test_len,) + image_size, dtype=np.uint8)
                group_test.create_dataset("labels", (test_len,), dtype=np.float32)
                group_test.create_dataset("dates", (test_len,), dtype=h5py.string_dtype(encoding='utf-8'))
                
            dates_test = list(group_test["dates"][:])
            
            # Get the last used index, for attachment
            last_used_index_test = len([date for date in dates_test if date != b''])
        
        # Get the unique identifiers for each permno's
        existing_dates = set(dates_train + dates_test)
        
    
        for day in range(len(df)-lookback):
    
            # Add the date to the image
            date_dt = df.iloc[day+lookback]['Date']
            date_str = date_dt.strftime('%Y-%m-%d') # Formats the date as "YYYY-MM-DD"
            
            # Check if the image has been generated already
            if date_str.encode('utf-8') in existing_dates:
                # print('Already Exists')
                continue
            
            # Initially start with the assumption that it is not training data
            train_img = False
            
            # Add 20 days to date_dt as the prediction to hit
            if (len(df)-lookback) - day > 20:
                date_with_offset = df.iloc[day+lookback + 20]['Date']
                if date_with_offset < cut_off_date_dt:
                    
                    # Check if the training datapoint has a return label, if not, then we can not use it
                    if np.isnan(df.iloc[day+lookback]['Ret20']):
                        continue
                    
                    # Else if return label exists, then continue the training
                    train_img = True
    
            if train_img or last_trading_days[:-1].isin([date_dt]).any():
                
                # # Define the image name
                # image_name = f'{date_str[:4]}/{permno}_{date_str}.png'
                
                # for year in range(1993, 2020):
                
                #     year_path = os.path.join(DATA_PATH, f'{save_path}/{year}')
                    
                #     if not os.path.exists(year_path):
                #         os.makedirs(year_path)
                
                price_slice = df[day:day+lookback][['Open', 'High', 'Low', 'Close', 'MA']].reset_index(drop=True)
                volume_slice = df[day:day+lookback][['Volume']].reset_index(drop=True)
                
                number_price_change = len(np.unique(price_slice.iloc[:,:4]))
                
                # # Check if the image has been generated already
                # check_path = os.path.join(DATA_PATH, f'{save_path}/{image_name}')
                # if not os.path.exists(check_path) or is_image_corrupt(check_path):
                
                    # price_slice = df[day:day+lookback][['Open', 'High', 'Low', 'Close', 'MA']].reset_index(drop=True)
                    # volume_slice = df[day:day+lookback][['Volume']].reset_index(drop=True)
        
                    # # number of no transactions days > 0.2*look back days
                    # if (1.0*(price_slice[['Open', 'High', 'Low', 'Close']].sum(axis=1)/price_slice['Open'] == 4)).sum() > lookback//5: 
                    #     print('No transactions')
                    #     invalid_dates.append(df.iloc[day]['Date']) # trading dates not surviving the validation
                    #     continue
                    
                # Check for missing data at the boundaries
                if pd.isna(price_slice.loc[0, ['High', 'Low']]).any() or pd.isna(price_slice.iloc[-1][['High', 'Low']]).any():
                    continue  # Skip this slice as the first or last entry is incomplete
                
                
                
                window_max = np.nanmax(price_slice.values)
                window_min = np.nanmin(price_slice.values)
                
                price_range = window_max - window_min
                
                if number_price_change < 14:
                    price_range = window_max - window_min
                    
                    padding = (14 - number_price_change)//2 * (price_range / number_price_change)
                    
                    window_min -= padding
                    window_max += padding
                    
                # project price and volume into quantiles
                price_slice = (price_slice - window_min)/(window_max - window_min)
                volume_slice = (volume_slice - np.nanmin(volume_slice.values))/(np.nanmax(volume_slice.values) - np.nanmin(volume_slice.values))
        
                # project price and volume into pixel
                if not show_volume:
                    price_slice = price_slice.apply(lambda x: x*(image_size[0]-2) + 1).astype(int)
                else:
                    # Define the volume allocation space on the graph
                    vol_space = int(0.2*(image_size[0]))
        
                    # Define the range of the price_slice
                    price_slice_lower_limit = vol_space + 1
                    price_slice_upper_limit = image_size[0] - 1
                    price_slice_range = price_slice_upper_limit - price_slice_lower_limit
        
                    # # Scale and shift the prices
                    # price_slice = price_slice.apply(lambda x: x*price_slice_range + price_slice_lower_limit).astype(int)
                    
                    # # Scale and shift the volume
                    # volume_slice = volume_slice.apply(lambda x: x*(vol_space-1) + 1).astype(int) # Setting the lowest volume limit to 1
                    
                    # Scale and shift the prices, handle NaN explicitly within the lambda
                    price_slice = price_slice.apply(lambda col: col.apply(lambda x: int(x * price_slice_range + price_slice_lower_limit) if pd.notna(x) else np.nan))
                    
                    # Scale and shift the volume, handle NaN explicitly within the lambda
                    volume_slice = volume_slice.apply(lambda col: col.apply(lambda x: (int(x * (vol_space - 1) + 1) if not pd.isna(x) else np.nan))) # Setting the lowest volume limit to 1
        
        
                # Create the image
                image = np.zeros(image_size)
                for i in range(len(price_slice)):
                    # # draw candlelist 
                    # image[price_slice.loc[i]['Open'], i*3] = 255
                    # image[price_slice.loc[i]['Low']:price_slice.loc[i]['High']+1, i*3+1] = 255
                    # image[price_slice.loc[i]['Close'], i*3+2] = 255
                    # # draw indicators
                    # for ind in ['MA']:
                    #     image[price_slice.loc[i][ind], i*3:i*3+2] = 255
                    # # draw volume bars
                    # if show_volume:
                    #     image[:volume_slice.loc[i]['Volume'], i*3+1] = 255
                    
                    # Draw price bar; handle missing values by checking if they are NaN
                    if not pd.isna(price_slice.at[i, 'High']) and not pd.isna(price_slice.at[i, 'Low']):
                        image[int(price_slice.at[i, 'Low']):int(price_slice.at[i, 'High']+1), i*3+1] = 255
                    if not pd.isna(price_slice.at[i, 'Open']):
                        image[int(price_slice.at[i, 'Open']), i*3] = 255
                    if not pd.isna(price_slice.at[i, 'Close']):
                        image[int(price_slice.at[i, 'Close']), i*3+2] = 255
                    
                    # draw indicators
                    for ind in ['MA']:
                        if not pd.isna(price_slice.loc[i][ind]):
                            image[int(price_slice.loc[i][ind]), i*3:i*3+2] = 255
        
                    # Draw volume if required
                    if show_volume:
                        if not pd.isna(volume_slice.at[i, 'Volume']):
                            image[:int(volume_slice.at[i, 'Volume']), i*3+1] = 255
            
            
                # Flip the y-axis of the image matrix, so we have a normal graph with volume at the bottom
                flipped_image = np.flipud(image)
        
                # image_name = f'{save_path}/{permno}_{date_str}.png'
                # image_name = f'{permno}_{date_str}.png'
        
                # Ensure the image is in 8-bit unsigned integer format
                image_8bit = np.clip(flipped_image, 0, 255).astype(np.uint8)
        
                # # Convert the matrix to a PIL image in 'L' mode (grayscale)
                # image_PIL = Image.fromarray(image_8bit, 'L')
        
                # # Save the image as a PNG file
                # image_PIL.save(os.path.join(DATA_PATH, f'{save_path}/{image_name}'))
                
                
                # # Convert date_str using the format YYYY-MM-DD
                # date_dt = datetime.strptime(date_str, '%Y-%m-%d')
                
                # # Convert cut_off_date using the format DD-MM-YYYY
                # cut_off_date_dt = datetime.strptime(cut_off_date, '%d-%m-%Y')
        
                if pred_move:
                    # Add the actual return label to the image
                    # label_ret5 = df.iloc[day+lookback]['Ret5'].round(3)
                    label_ret20 = df.iloc[day+lookback]['Ret20'].round(3)
                    # label_ret60 = df.iloc[day+lookback]['Ret60'].round(3)
                else:
                    # Add the return direction label to the image, 1 if positive, 0 if negative
                    # label_ret5 = 1 if df.iloc[day+lookback]['Ret5'] > 0 else 0
                    label_ret20 = 1 if df.iloc[day+lookback]['Ret20'] > 0 else 0
                    # label_ret60 = 1 if df.iloc[day+lookback]['Ret60'] > 0 else 0
                    
                    
                if train_img:
                    group_train["images"][last_used_index_train] = image_8bit.reshape(image_size)
                    group_train["labels"][last_used_index_train] = label_ret20
                    group_train["dates"][last_used_index_train] = date_str
                    
                    # Increment index
                    last_used_index_train += 1
                else:
                    group_test["images"][last_used_index_test] = image_8bit.reshape(image_size)
                    group_test["labels"][last_used_index_test] = label_ret20
                    group_test["dates"][last_used_index_test] = date_str
                    
                    # Increment index
                    last_used_index_test += 1
        
                # if pred_move:
                #      if train_img:
                #         # Add the actual return label to the image
                #         # label_ret5 = df.iloc[day+lookback]['Ret5'].round(3)
                #         label_ret20 = df.iloc[day+lookback]['Ret20'].round(3)
                #         # label_ret60 = df.iloc[day+lookback]['Ret60'].round(3)
            
                #         # entry = [image_name, label_ret20]
                #         # dataset.append(entry)
                        
                        # group_test["images"][index] = image_8bit
                        # group_test["labels"][index] = label_ret20
                        # group_test["dates"][index] = date_str
                        
                #     if date_dt >= cut_off_date_dt and last_trading_days[:-1].isin([date_dt]).any():
                #         # Get the index of the current last trading day
                #         current_index = df[df['Date'] == date_dt].index[0]
    
                #         # Calculate the index for the next month's last trading day
                #         next_month_date = last_trading_days[last_trading_days > date_dt].iloc[0]
                #         next_month_index = df[df['Date'] == next_month_date].index[0]
    
                #         # Calculate return from current to next month's last trading day
                #         return_next_month = (df.at[next_month_index, 'Close'] - df.at[current_index, 'Close']) / df.at[current_index, 'Close']
    
                #         # Ensure that the image exists
                #         if current_index - lookback >= 0:
                            
                #             # Append to test_dataset
                #             # test_dataset.append([image_name, return_next_month.round(3)])
                            
                #             group_test["images"][index] = image_8bit
                #             group_test["labels"][index] = return_next_month.round(3)
                #             group_test["dates"][index] = date_str
                # else:
                #     if train_img:
                        # # Add the return direction label to the image, 1 if positive, 0 if negative
                        # # label_ret5 = 1 if df.iloc[day+lookback]['Ret5'] > 0 else 0
                        # label_ret20 = 1 if df.iloc[day+lookback]['Ret20'] > 0 else 0
                        # # label_ret60 = 1 if df.iloc[day+lookback]['Ret60'] > 0 else 0
            
                #         entry = [image_name, label_ret20]
                #         dataset.append(entry)
                #     if date_dt >= cut_off_date_dt and last_trading_days[:-1].isin([date_dt]).any():
                #         # Get the index of the current last trading day
                #         current_index = df[df['Date'] == date_dt].index[0]
    
                #         # Calculate the index for the next month's last trading day
                #         next_month_date = last_trading_days[last_trading_days > date_dt].iloc[0]
                #         next_month_index = df[df['Date'] == next_month_date].index[0]
    
                #         # Calculate return from current to next month's last trading day
                #         return_next_month = 1 if (df.at[next_month_index, 'Close'] - df.at[current_index, 'Close']) / df.at[current_index, 'Close'] > 0 else 0
                        
                #         # Ensure that the image exists
                #         if current_index - lookback >= 0:
                            
                #             # Append to test_dataset
                #             test_dataset.append([image_name, return_next_month])


    return #dataset, test_dataset



def show_single_graph(path, idx, run=None):

    # # Load the image from file
    # image_path = f'{DATA_PATH}/{path}/{entry[0]}'
    # # image_path = f'{entry[0]}'
    # image = Image.open(image_path)

    # # Split the string and extract the date
    # date_part = entry[0].split('_')[1].split('.')[0]
    
    with h5py.File(path, 'r') as file:
        binary_image = file["images"][idx]
        label = file["labels"][idx]
        date = file["dates"][idx].decode('utf-8')
    
    # Convert binary to image
    image = Image.open(io.BytesIO(binary_image)).convert('L')
    
    # Convert image to numpy array
    image_array = np.array(image)
    
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
        run["plots/show_single_graph"].upload(tmpfile.name)
        plt.close(plt.gcf())
    
    else:
        plt.show()



class GraphDataset():
    def __init__(self, df, win_size, mode, label, market, indicator = [], show_volume=False, predict_movement=False, cut_off_date = None, parallel_num=-1):

        assert win_size in [5, 20, 60], f'Wrong look back days: {win_size}'
        assert mode in ['train', 'test', 'inference'], f'Type Error: {mode}'
        assert label in ['Ret5', 'Ret20', 'Ret60'], f'Wrong Label: {label}'
        # assert market in ['SP500', 'US']

        self.df = df
        self.mode = mode
        self.label = label
        self.market = market
        self.indicator = indicator
        self.show_volume = show_volume
        self.pred_move = predict_movement
        self.parallel_num = parallel_num
        self.cut_off_date = cut_off_date
        

        # Define the image size, as each day needs to be represented by 3 pixels
        if win_size == 5:
            self.image_size = (32, 15)
        elif win_size == 20:
            self.image_size = (64, 60)
        elif win_size == 60:
            self.image_size = (96, 180)

        self.window_size = win_size


    # Generate all the images for all the stocks
    def generate_images(self):

        # Save the images
        if self.show_volume:
            table_name = f'I{self.window_size}VolTInd{list(self.indicator[0].values())[0]}'
        else:
            table_name = f'I{self.window_size}VolFInd{list(self.indicator[0].values())[0]}'

        # Check if the directory exists, if not, create it
        if not os.path.exists(os.path.join(DATA_PATH, table_name)):
            os.makedirs(os.path.join(DATA_PATH, table_name))
            
            
        # Setup the path to the datasets
        hdf5_train_path = f'{DATA_PATH}/{self.market}/{table_name}/Initial_h5/'
        hdf5_test_path = f'{DATA_PATH}/{self.market}/{table_name}Initial_h5/'
        
        # Open both HDF5 files, one for training data and one for testing data
        with h5py.File(hdf5_train_path, 'a') as hdf5_file_train, h5py.File(hdf5_test_path, 'a') as hdf5_file_test:
            
            # Ensure the datasets exist and are resizable
            for hdf5_file in [hdf5_file_train, hdf5_file_test]:
                
                total_len = 10000
                
                if 'permnos' not in hdf5_file:
                    # hdf5_file.create_dataset("images", (total_len,), dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
                    # hdf5_file.create_dataset("labels", (total_len,), dtype=np.float32)
                    # hdf5_file.create_dataset("permnos", (total_len,), dtype=h5py.string_dtype(encoding='utf-8'))
                    # hdf5_file.create_dataset("dates", (total_len,), dtype=h5py.string_dtype(encoding='utf-8'))
                    
                    hdf5_file.create_dataset("permnos", (total_len,), dtype=h5py.string_dtype(encoding='utf-8'))
            
            
        # # Get permnos and dates from both train and test HDF5 files
        # train_permnos = hdf5_file_train["permnos"][:]
        # train_dates = hdf5_file_train["dates"][:]
        
        # test_permnos = hdf5_file_test["permnos"][:]
        # test_dates = hdf5_file_test["dates"][:]
        
        # # Combine permnos and dates into tuples for train and test
        # train_identifiers = list(zip(train_permnos, train_dates))
        # test_identifiers = list(zip(test_permnos, test_dates))
        
        # # Combine the two lists into one
        # identifiers = train_identifiers + test_identifiers
        
        # Utilize parallel processing to generate the images for all the stocks quickly
        dataset_all = Parallel(n_jobs=self.parallel_num)(delayed(image_generator)(
                                        g[0], g[1], image_size = self.image_size, 
                                        lookback = self.window_size,
                                        indicator = self.indicator, 
                                        show_volume = self.show_volume,
                                        pred_move = self.pred_move,
                                        save_path = table_name,
                                        train_file = hdf5_file_train,
                                        test_file = hdf5_file_test,
                                        mode = self.mode,
                                        cut_off_date = self.cut_off_date
                                        ) for g in tqdm(self.df.groupby('permno'), 
                                                        desc=f'Generating Images'))
                                                        
        ### Tester ###
        
        # filtered_data = sp500_daily.groupby('permno').filter(lambda x: len(x) >= 200)
        
        # Utilize parallel processing to generate the images for all the stocks quickly
        dataset_all = Parallel(n_jobs=parallel_num)(delayed(image_generator)(
                                        g[0], g[1], image_size, 
                                        lookback,
                                        indicator, 
                                        show_volume,
                                        pred_move,
                                        save_path,
                                        hdf5_train_path,
                                        hdf5_test_path,
                                        mode,
                                        cut_off_date
                                        ) for g in tqdm(df.groupby('permno'), 
                                                        desc=f'Generating Images'))
                                                        
        #############

        # # Concatenate the dataset
        # image_set_train = []
        # image_set_test = []
        # for data in dataset_all:
        #     # for row in data:
        #     #     image_set.append(row)
        #     for row_tr in data[0]:
        #         image_set_train.append(row_tr)
                
        #     for row_te in data[1]:    
        #         image_set_test.append(row_te)
        
        # # Clear memory
        # dataset_all = []
        
        # # Convert image_set to a DataFrame
        # image_set_train = pd.DataFrame(image_set_train, columns=['file_name', 'ret'])
        # image_set_test = pd.DataFrame(image_set_test, columns=['file_name', 'ret'])
        
        
        # # Define file paths
        # train_path = f'{DATA_PATH}/{self.market}/{table_name}_dataset_train.csv'
        # test_path = f'{DATA_PATH}/{self.market}/{table_name}_dataset_test.csv'
        
        # # Check for the existence of files and append new data if they exist
        # def append_new_data(file_path, new_data):
        #     if os.path.exists(file_path):
                
        #         # Read existing data
        #         existing_data = pd.read_csv(file_path)
                
        #         # Combine old and new data
        #         combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                
        #         # Drop duplicates based on the first column
        #         clean_data = combined_data.drop_duplicates(subset='file_name', keep='first')
                
        #         # Save back to csv
        #         clean_data.to_csv(file_path, index=False)
                
        #         return clean_data
        #     else:
        #         # If file does not exist, simply write the new data
        #         new_data.to_csv(file_path, index=False)
                
        #         return new_data
        
        # # Apply the function to train and test datasets
        # if len(image_set_train) > 0:
        #     image_set_train = append_new_data(train_path, image_set_train)
        # if len(image_set_test) > 0:
        #     image_set_test = append_new_data(test_path, image_set_test)

        # return image_set_train, image_set_test, table_name

    


a = list(df.groupby('permno'))[:10]

for i, g in enumerate(a, start=1):  # start=1 makes the iteration count start from 1
    print(f"Iteration {i} of {len(a)}")
    b = image_generator(
        g[0], g[1], image_size, 
        lookback,
        indicator, 
        show_volume,
        pred_move,
        save_path,
        hdf5_file_train,
        hdf5_file_test,
        mode,
        cut_off_date
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    