from project_imports import *


# Description: This file contains the functions to generate the graphs for each stock


  
def image_generator(permno, df, image_size, lookback, indicator, show_volume, drop_rate, 
                    adj_prc, pred_move, existing_dates, mode, dates_to_gen):
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
    
    # Remove any leading NaN's with missing close price
    df = df[~df['Close'].ffill().isna()]
    
    # Get moving average
    if len(indicator) > 0:
        for ind in indicator:
            for key, value in ind.items():
                df = df.assign(**{key: df['Close'].rolling(window=value).mean()})
                
                # Drop the starting NA's as they will no longer have any data on MA
                df = df[value-1:]
             
                
    # Remove rows with NA's
    # df = df.dropna()
    
    # Convert to date
    # cut_off_date_dt = pd.to_datetime(cut_off_date, format='%d-%m-%Y')
    
    # # Split data to train and test data
    # df_train = df[df['Date'] <= cut_off_date_dt]
    # df_test = df[df['Date'] > cut_off_date_dt]
    
    # # Prepare the datasets 
    dataset = []
    # test_dataset = []
    
    # Skip the generation if the DataFrame has fewer than 40 entries
    if df.shape[0] < lookback:
        return dataset #, test_dataset  # Exit the function early
    
    # Group by year and month to find the last trading day
    last_trading_days = df.groupby([df['Date'].dt.year, df['Date'].dt.month])['Date'].max()
    
    # last_trading_days = last_trading_days[last_trading_days>=cut_off_date_dt]
        
    # Combine permnos and dates into a single string for fast comparison using a set
    # existing_combined_set = set(np.char.add(existing_permnos.astype(str), existing_dates.astype(str)))
    # existing_combined_set = set(zip(existing_permnos, existing_dates))

    
    for day in range(len(df)-lookback):

        # Add the date to the image
        date_dt = df.iloc[day+lookback]['Date']
        date_str = date_dt.strftime('%Y-%m-%d').encode('utf-8') # Formats the date as "YYYY-MM-DD"
        
        permno_str = str(permno).encode('utf-8')
        
        # Combine permno and date into a single string
        # combined_str = permno_str + date_str

        # Check if the image has already been generated
        if date_str in existing_dates:
            continue  # Skip if the image already exists
        # print('new')
        # # Check if the image has been generated already
        # if permno_str in existing_permnos and date_str in existing_dates:
        #     # print('Already Exists')
        #     continue
        
        # Initially start with the assumption that it is not training data
        # train_img = False
        
        # # Add 20 days to date_dt as the prediction to hit
        # if (len(df)-lookback) - day > 20:
        #     date_with_offset = df.iloc[day+lookback + 20]['Date']
        #     if date_with_offset < cut_off_date_dt:
                
        #         # Check if the training datapoint has a return label, if not, then we can not use it
        #         if np.isnan(df.iloc[day+lookback]['Ret20']):
        #             continue
                
        #         # Add a random dropout check to decide if this entry should be skipped
        #         if random.uniform(0, 1) < drop_rate:
        #             continue
                
        #         # Else if return label exists, then continue the training
        #         train_img = True

        if date_dt in dates_to_gen:
            
            price_slice = df[day:day+lookback][['Open', 'High', 'Low', 'Close', 'MA', 'ret', 'Date']].reset_index(drop=True)
            volume_slice = df[day:day+lookback][['Volume']].reset_index(drop=True)
            
            # Define the specific price columns to check for NaN values
            price_cols = ['Open', 'High', 'Low', 'Close', 'MA']
            
            # Calculate the percentage of NaN values in the selected columns
            missing_percentage = price_slice[price_cols].isna().mean().mean()
            
            # Check if more than a certain percentage of the data points are missing
            if missing_percentage > 0.3:
                continue  # Skip this slice
    
            if adj_prc:
                @staticmethod
                def adjust_price(df):
                    if len(df) == 0:
                        raise ValueError("adjust_price: Empty Dataframe")
                    if len(df.Date.unique()) != len(df):
                        raise ValueError("adjust_price: Dates not unique")
                    df = df.reset_index(drop=True)
            
                    fd_close = abs(df.at[0, "Close"])
                    if df.at[0, "Close"] == 0.0 or pd.isna(df.at[0, "Close"]):
                        raise ValueError("adjust_price: First day close is nan or zero")
            
                    pre_close = fd_close
                    res_df = df.copy()
            
                    res_df.at[0, "Close"] = 1.0
                    res_df.at[0, "Open"] = abs(res_df.at[0, "Open"]) / pre_close
                    res_df.at[0, "High"] = abs(res_df.at[0, "High"]) / pre_close
                    res_df.at[0, "Low"] = abs(res_df.at[0, "Low"]) / pre_close
                    res_df.at[0, "MA"] = abs(res_df.at[0, "MA"]) / pre_close
            
                    pre_close = 1
                    for i in range(1, len(res_df)):
                        today_closep = abs(res_df.at[i, "Close"])
                        today_openp = abs(res_df.at[i, "Open"])
                        today_highp = abs(res_df.at[i, "High"])
                        today_lowp = abs(res_df.at[i, "Low"])
                        today_ma = abs(res_df.at[i, "MA"])
                        today_ret = np.float64(res_df.at[i, "ret"])
            
                        res_df.at[i, "Close"] = (1 + today_ret) * pre_close
                        res_df.at[i, "Open"] = res_df.at[i, "Close"] / today_closep * today_openp
                        res_df.at[i, "High"] = res_df.at[i, "Close"] / today_closep * today_highp
                        res_df.at[i, "Low"] = res_df.at[i, "Close"] / today_closep * today_lowp
                        res_df.at[i, "MA"] = res_df.at[i, "Close"] / today_closep * today_ma
                        res_df.at[i, "ret"] = today_ret
            
                        if not pd.isna(res_df.at[i, "Close"]):
                            pre_close = res_df.at[i, "Close"]
            
                    return res_df
                
            
                price_slice = adjust_price(price_slice)
            
            price_slice = price_slice[['Open', 'High', 'Low', 'Close', 'MA']]
            
            # number_price_change = len(np.unique(price_slice.iloc[:,:4]))
            
            window_max = np.nanmax(price_slice.values)
            window_min = np.nanmin(price_slice.values)
            
            # price_range = window_max - window_min
            
            # if number_price_change < 14:
            #     price_range = window_max - window_min
              
            #     padding = (14 - number_price_change)//2 * (price_range / number_price_change)
                
            #     window_min -= padding
            #     window_max += padding
                
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
    
                # Scale and shift the prices, handle NaN explicitly within the lambda
                price_slice = price_slice.apply(lambda col: col.apply(lambda x: int(x * price_slice_range + price_slice_lower_limit) if pd.notna(x) else np.nan))
                
                # Scale and shift the volume, handle NaN explicitly within the lambda
                volume_slice = volume_slice.apply(lambda col: col.apply(lambda x: (int(x * (vol_space - 1) + 1) if not pd.isna(x) else np.nan))) # Setting the lowest volume limit to 1
    
    
    
            # from scipy.ndimage import map_coordinates

            # # Function to draw a line between two points in a numpy array
            # @staticmethod
            # def draw_line(image, x0, y0, x1, y1, color=255):
            #     """Draws a line between two points (x0, y0) and (x1, y1) in the image."""
            #     dx = abs(x1 - x0)
            #     dy = abs(y1 - y0)
            #     sx = 1 if x0 < x1 else -1
            #     sy = 1 if y0 < y1 else -1
            #     err = dx - dy
            
            #     while True:
            #         image[y0, x0] = color  # Set the pixel value
            #         if x0 == x1 and y0 == y1:
            #             break
            #         e2 = 2 * err
            #         if e2 > -dy:
            #             err -= dy
            #             x0 += sx
            #         if e2 < dx:
            #             err += dx
            #             y0 += sy
    
            # Create the image
            image = np.zeros(image_size)
            
            # Store the x and y coordinates of the `MA` points
            ma_points = []

            for i in range(len(price_slice)):
                
                # Draw price bar; handle missing values by checking if they are NaN
                if not pd.isna(price_slice.at[i, 'High']) and not pd.isna(price_slice.at[i, 'Low']):
                    image[int(price_slice.at[i, 'Low']):int(price_slice.at[i, 'High']+1), i*3+1] = 255
                if not pd.isna(price_slice.at[i, 'Open']):
                    image[int(price_slice.at[i, 'Open']), i*3] = 255
                if not pd.isna(price_slice.at[i, 'Close']):
                    image[int(price_slice.at[i, 'Close']), i*3+2] = 255
                
                # # draw indicators
                # for ind in ['MA']:
                #     if not pd.isna(price_slice.loc[i][ind]):
                #         image[int(price_slice.loc[i][ind]), i*3:i*3+2] = 255
                
                # Draw the MA as a single point in the center (i*3+1)
                if not pd.isna(price_slice.at[i, 'MA']):
                    ma_y = int(price_slice.at[i, 'MA'])
                    ma_x = i*3 + 1  # Center of the 3-pixel block
                    image[ma_y, ma_x] = 255
                    ma_points.append((ma_x, ma_y))  # Store the MA point for line drawing later
    
                # Draw volume if required
                if show_volume:
                    if not pd.isna(volume_slice.at[i, 'Volume']):
                        image[:int(volume_slice.at[i, 'Volume']), i*3+1] = 255
        
            
            # # Step 2: Connect the MA points with a line
            # for j in range(len(ma_points) - 1):
            #     # Get coordinates of consecutive MA points
            #     x0, y0 = ma_points[j]
            #     x1, y1 = ma_points[j + 1]
            
            #     # Draw a line between the consecutive points
            #     draw_line(image, x0, y0, x1, y1)
            
            # Step 3: Connect the MA points using OpenCV's line drawing
            for j in range(len(ma_points) - 1):
                # Get coordinates of consecutive MA points
                x0, y0 = ma_points[j]
                x1, y1 = ma_points[j + 1]
            
                # Draw a line between the consecutive points using OpenCV
                cv2.line(image, (x0, y0), (x1, y1), color=255, thickness=1)
        
            # Flip the y-axis of the image matrix, so we have a normal graph with volume at the bottom
            flipped_image = np.flipud(image)
    
            # image_name = f'{save_path}/{permno}_{date_str}.png'
            # image_name = f'{permno}_{date_str}.png'
    
            # Ensure the image is in 8-bit unsigned integer format
            image_8bit = np.clip(flipped_image, 0, 255).astype(np.uint8)
            
            # # Create a PIL image from the numpy array
            # image = Image.fromarray(image_8bit, 'L')
            
            # # Plot the image using matplotlib
            # plt.imshow(image, cmap='gray')
            # plt.axis('off')  # Hide the axis
            # plt.title("Grayscale Image")
            # plt.show()
            
            # Using 'with' to ensure the buffer is properly handled
            with io.BytesIO() as buffer:
                # Convert the NumPy array to a Pillow image and save as PNG to the buffer
                Image.fromarray(image_8bit).save(buffer, format="PNG")
                # Get the PNG compressed binary data
                binary_image = buffer.getvalue()
            
            
            compressed_image = np.frombuffer(binary_image, dtype=np.uint8)
    
            # # Convert the matrix to a PIL image in 'L' mode (grayscale)
            # image_PIL = Image.fromarray(image_8bit, 'L')
    
            # # Save the image as a PNG file
            # image_PIL.save(os.path.join(DATA_PATH, f'{save_path}/{image_name}'))
    
            # if pred_move:
            #     # Add the actual return label to the image
            #     # label_ret5 = df.iloc[day+lookback]['Ret5'].round(3)
            #     label_ret20 = df.iloc[day+lookback]['Ret20'].round(6)
            #     # label_ret60 = df.iloc[day+lookback]['Ret60'].round(3)
            # else:
            #     # Add the return direction label to the image, 1 if positive, 0 if negative
            #     # label_ret5 = 1 if df.iloc[day+lookback]['Ret5'] > 0 else 0
            #     label_ret20 = 1 if df.iloc[day+lookback]['Ret20'] > 0 else 0
            #     # label_ret60 = 1 if df.iloc[day+lookback]['Ret60'] > 0 else 0
                
            # Define a helper function to validate if the entry is not empty
            def is_valid_entry(image, permno, date):
                # Check that none of the entries are None or empty
                if image is None or len(image) == 0:
                    return False
                # if (label is None or label == '' ) and train_img:
                #     return False
                if permno is None or permno == '':
                    return False
                if date is None or date == '':
                    return False
                return True
                
               
            # Check that all fields are valid before appending to the test dataset
            if is_valid_entry(compressed_image, permno_str, date_str):
                # If valid, add to test dataset
                # # Step 1: Clip the returns to a minimum of -1
                # label_ret20 = label_ret20 if label_ret20 > -1 else -1
                
                # # Step 2: Replace NaN values with -99
                # label_ret20 = np.nan_to_num(label_ret20, nan=-99)
                
                entry = [compressed_image, permno_str, date_str]
            else:
                continue
                
            # if train_img:
                
            #     # entry = [image_8bit.reshape(image_size), label_ret20, permno_str, date_str]
            #     # entry = [compressed_image, label_ret20, permno_str, date_str]
            #     dataset.append(entry)
            # else:
                
            #     # entry = [compressed_image, label_ret20, permno_str, date_str]
                
            #     test_dataset.append(entry)


            dataset.append(entry)
            
    return dataset #, test_dataset




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
        permno = file["permnos"][idx].decode('utf-8')
        date = file["dates"][idx].decode('utf-8')
    
    # Convert binary to image
    image = Image.open(io.BytesIO(binary_image)).convert('L')
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    plt.figure()
    plt.imshow(image_array, cmap=plt.get_cmap('gray'))
    plt.title(f'Ret: {label:.3f}\n Last Date is: {date}\n Permno is: {permno}')
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
        
        

def append_to_hdf5(dataset_all, h5_path_train, h5_path_test):
    """
    Append the generated dataset to an HDF5 file for both training and testing data.
    This function ensures that there are no duplicate rows with the same combination
    of permno and date, as they form a unique identifier.

    Args:
        dataset_all: The full dataset containing both train and test data.
        h5_path: Path to the HDF5 file (train or test).
    """

    # Initialize empty lists to store data
    train_images, train_labels, train_permnos, train_dates = [], [], [], []
    test_images, test_labels, test_permnos, test_dates = [], [], [], []

    # Flatten the dataset_all
    for data in dataset_all:
        for entry in data[0]:
            if len(entry) > 0:  # Ensure train data exists
                train_images.append(entry[0])  # image
                train_labels.append(entry[1])  # label
                train_permnos.append(entry[2])  # permno
                train_dates.append(entry[3])  # date
                    
        for entry in data[1]:
            if len(entry) > 0:  # Ensure test data exists
                test_images.append(entry[0])  # image
                test_labels.append(entry[1])  # label
                test_permnos.append(entry[2])  # permno
                test_dates.append(entry[3])  # date

    # # Convert lists to numpy arrays
    # train_images = np.array(train_images, dtype=np.uint8)
    # train_labels = np.array(train_labels, dtype=np.float32)
    # train_permnos = np.array(train_permnos, dtype='S10')  # UTF-8 strings for permnos
    # train_dates = np.array(train_dates, dtype='S10')  # UTF-8 strings for dates
    
    # Convert the lists to numpy arrays
    train_images = np.array(train_images, dtype=h5py.special_dtype(vlen=np.uint8))
    train_labels = np.array(train_labels, dtype=np.float32)
    train_permnos = np.array(train_permnos, dtype=h5py.string_dtype(encoding='utf-8'))
    train_dates = np.array(train_dates, dtype=h5py.string_dtype(encoding='utf-8'))

    test_images = np.array(test_images, dtype=h5py.special_dtype(vlen=np.uint8))
    test_labels = np.array(test_labels, dtype=np.float32)
    test_permnos = np.array(test_permnos, dtype=h5py.string_dtype(encoding='utf-8'))  # UTF-8 strings for permnos
    test_dates = np.array(test_dates, dtype=h5py.string_dtype(encoding='utf-8'))  # UTF-8 strings for dates


    if len(train_images) > 0:
        # Save train data
        with h5py.File(h5_path_train, 'a') as h5_file:
            
            # Check if datasets already exist, otherwise create them
            if "images" not in h5_file:
                h5_file.create_dataset('images', data=train_images, maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
                h5_file.create_dataset('labels', data=train_labels, maxshape=(None,), dtype=np.float32)
                h5_file.create_dataset('permnos', data=train_permnos, maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
                h5_file.create_dataset('dates', data=train_dates, maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
            else:
                # Load existing data
                existing_permnos = h5_file["permnos"][:]
                existing_dates = h5_file["dates"][:]
                
                # Identify new rows (that do not exist in the file)
                # unique_indices = []
                # for i in range(len(train_permnos)):
                #     if not ((train_permnos[i] in existing_permnos) and (train_dates[i] in existing_dates)):
                #         unique_indices.append(i)
                        
                # for i in range(len(train_permnos)):
                #     if not (set(zip(train_permnos[i], train_dates[i])) in set(zip(existing_permnos, existing_dates))):
                #         unique_indices.append(i)
                        
                
                # Combine permnos and dates into a single string for fast comparison using a set
                existing_combined_set = set(np.char.add(existing_permnos.astype(str), existing_dates.astype(str)))
                
                # Combine new permnos and dates into a single string
                new_combined = np.char.add(train_permnos.astype(str), train_dates.astype(str))
                
                # Identify new rows that do not exist in the set of existing rows
                unique_indices = [i for i, value in enumerate(new_combined) if value not in existing_combined_set]

                
                
                
                
                # Filter new data to append (avoid duplicates)
                train_images = train_images[unique_indices]
                train_labels = train_labels[unique_indices]
                train_permnos = train_permnos[unique_indices]
                train_dates = train_dates[unique_indices]
    
                # Append only unique data
                if len(train_images) > 0:
                    h5_file["images"].resize((h5_file["images"].shape[0] + train_images.shape[0]), axis=0)
                    h5_file["images"][-train_images.shape[0]:] = train_images
    
                    h5_file["labels"].resize((h5_file["labels"].shape[0] + train_labels.shape[0]), axis=0)
                    h5_file["labels"][-train_labels.shape[0]:] = train_labels
    
                    h5_file["permnos"].resize((h5_file["permnos"].shape[0] + train_permnos.shape[0]), axis=0)
                    h5_file["permnos"][-train_permnos.shape[0]:] = train_permnos
    
                    h5_file["dates"].resize((h5_file["dates"].shape[0] + train_dates.shape[0]), axis=0)
                    h5_file["dates"][-train_dates.shape[0]:] = train_dates
    
    print(f"Appended {len(train_labels)} new rows to training data")
    
    if len(test_images) > 0:
        # Save test data
        with h5py.File(h5_path_test, 'a') as h5_file:
            # Check if datasets already exist, otherwise create them
            if "images" not in h5_file:
                
                h5_file.create_dataset('images', data=test_images, maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
                h5_file.create_dataset('labels', data=test_labels, maxshape=(None,), dtype=np.float32)
                h5_file.create_dataset('permnos', data=test_permnos, maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
                h5_file.create_dataset('dates', data=test_dates, maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
            else:
                # Load existing data
                existing_permnos = h5_file["permnos"][:]
                existing_dates = h5_file["dates"][:]
                
                # Identify new rows (that do not exist in the file)
                # unique_indices = []
                # for i in range(len(test_permnos)):
                #     if not ((test_permnos[i] in existing_permnos) and (test_dates[i] in existing_dates)):
                #         unique_indices.append(i)
                
                # Combine permnos and dates into a single string for fast comparison using a set
                existing_combined_set = set(np.char.add(existing_permnos.astype(str), existing_dates.astype(str)))
                
                # Combine new permnos and dates into a single string
                new_combined = np.char.add(test_permnos.astype(str), test_dates.astype(str))
                
                # Identify new rows that do not exist in the set of existing rows
                unique_indices = [i for i, value in enumerate(new_combined) if value not in existing_combined_set]
                
                # Filter new data to append (avoid duplicates)
                test_images = test_images[unique_indices]
                test_labels = test_labels[unique_indices]
                test_permnos = test_permnos[unique_indices]
                test_dates = test_dates[unique_indices]
    
                # Append only unique data
                if len(test_images) > 0:
                    h5_file["images"].resize((h5_file["images"].shape[0] + test_images.shape[0]), axis=0)
                    h5_file["images"][-test_images.shape[0]:] = test_images
    
                    h5_file["labels"].resize((h5_file["labels"].shape[0] + test_labels.shape[0]), axis=0)
                    h5_file["labels"][-test_labels.shape[0]:] = test_labels
    
                    h5_file["permnos"].resize((h5_file["permnos"].shape[0] + test_permnos.shape[0]), axis=0)
                    h5_file["permnos"][-test_permnos.shape[0]:] = test_permnos
    
                    h5_file["dates"].resize((h5_file["dates"].shape[0] + test_dates.shape[0]), axis=0)
                    h5_file["dates"][-test_dates.shape[0]:] = test_dates

    print(f"Appended {len(test_labels)} new rows to test data")



class GraphDataset():
    def __init__(self, df, win_size, mode, label, market, indicator = [], show_volume=False, 
                 drop_rate=0.0, adj_prc=True, predict_movement=False, dates_to_gen = None, parallel_num=-1):

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
        self.drop_rate = drop_rate
        self.adj_prc = adj_prc
        self.pred_move = predict_movement
        self.parallel_num = parallel_num
        self.dates_to_gen = dates_to_gen
        

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
        hdf5_train_path = f'{DATA_PATH}/{self.market}/{table_name}_dataset.h5'
        # hdf5_test_path = f'{DATA_PATH}/{self.market}/{table_name}_test.h5'
        
        # # Open both HDF5 files, one for training data and one for testing data
        # with h5py.File(hdf5_train_path, 'a') as hdf5_file_train, h5py.File(hdf5_test_path, 'a') as hdf5_file_test:
            
        #     # Ensure the datasets exist and are resizable
        #     for hdf5_file in [hdf5_file_train, hdf5_file_test]:
                
        #         total_len = 10000
                
        #         if 'permnos' not in hdf5_file:
        #             # hdf5_file.create_dataset("images", (total_len,), dtype=h5py.special_dtype(vlen=np.dtype('uint8')))
        #             # hdf5_file.create_dataset("labels", (total_len,), dtype=np.float32)
        #             # hdf5_file.create_dataset("permnos", (total_len,), dtype=h5py.string_dtype(encoding='utf-8'))
        #             # hdf5_file.create_dataset("dates", (total_len,), dtype=h5py.string_dtype(encoding='utf-8'))
                    
        #             hdf5_file.create_dataset("permnos", (total_len,), dtype=h5py.string_dtype(encoding='utf-8'))
            
            
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
        
        with h5py.File(hdf5_train_path, 'a') as h5_file:
            # Check if datasets already exist, otherwise continue
            if "images" in h5_file:
                # Load existing data
                existing_permnos = h5_file["permnos"][:]
                existing_dates = h5_file["dates"][:]
            else:
                existing_permnos = np.array([], dtype=h5py.string_dtype(encoding='utf-8'))
                existing_dates = np.array([], dtype=h5py.string_dtype(encoding='utf-8'))
        
        
        # with h5py.File(hdf5_test_path, 'a') as h5_file:
        #     # Check if datasets already exist, otherwise continue
        #     if "images" in h5_file:
        #         # Load existing data
        #         existing_permnos_test = h5_file["permnos"][:]
        #         existing_dates_test = h5_file["dates"][:]
        #     else:
        #         existing_permnos_test = np.array([], dtype=h5py.string_dtype(encoding='utf-8'))
        #         existing_dates_test = np.array([], dtype=h5py.string_dtype(encoding='utf-8'))
        
        # # Now concatenate the train and test arrays
        # existing_permnos = np.concatenate([existing_permnos_train, existing_permnos_test])
        # existing_dates = np.concatenate([existing_dates_train, existing_dates_test])
        
        
        
        # test = existing_dates[existing_permnos == b'100597']
        
        # batch_size = 16 * 10  # Define a manageable batch size, cores x scalar

        # groups = list(df.groupby('permno'))  # Convert groupby object to a list
        # for i in range(0, len(groups), batch_size):
        #     batch = groups[i:i + batch_size]
        
        # existing_dict = {p.decode('utf-8'): set(d) for p, d in zip(existing_permnos, existing_dates)}

        
        # Utilize parallel processing to generate the images for all the stocks quickly
        dataset_all = Parallel(n_jobs=self.parallel_num)(delayed(image_generator)(
                                        g[0], g[1], image_size = self.image_size, 
                                        lookback = self.window_size,
                                        indicator = self.indicator, 
                                        show_volume = self.show_volume,
                                        drop_rate = self.drop_rate,
                                        adj_prc = self.adj_prc,
                                        pred_move = self.pred_move,
                                        # existing_dates = existing_dict.get(str(g[0]), set()),
                                        existing_dates = set(existing_dates[existing_permnos == str(g[0]).encode('utf-8')]),
                                        mode = self.mode,
                                        dates_to_gen = self.dates_to_gen
                                        ) for g in tqdm(self.df.groupby('permno'), 
                                                        desc=f'Generating Images'))
                                                            
            ### Tester ###
            
            # filtered_data = sp500_daily.groupby('permno').filter(lambda x: len(x) >= 200)
            
        # # Utilize parallel processing to generate the images for all the stocks quickly
        # dataset_all = Parallel(n_jobs=parallel_num)(delayed(image_generator)(
        #                                 g[0], g[1], image_size, 
        #                                 lookback,
        #                                 indicator, 
        #                                 show_volume,
        #                                 drop_rate,
        #                                 adj_prc,
        #                                 pred_move,
        #                                 set(existing_dates[existing_permnos == str(g[0]).encode('utf-8')]),
        #                                 mode,
        #                                 dates_to_gen
        #                                 ) for g in tqdm(df.groupby('permno'), 
        #                                                 desc=f'Generating Images'))
                                                        
        # tqdm(batch, 
        #                 desc=f'Generating Images Batch {i//batch_size+1} / {len(groups)//batch_size+1}'))
                        
        # Append data to train and test HDF5 files
        # append_to_hdf5(dataset_all, hdf5_train_path, hdf5_test_path)
                                                        
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
        return dataset_all

    


# a = list(df.groupby('permno'))[929:930]

# empty_list = []

# for i, g in enumerate(a, start=1):  # start=1 makes the iteration count start from 1
#     print(f"Iteration {i} of {len(a)}")
#     b = image_generator(
#         g[0], g[1], image_size, 
#         lookback,
#         indicator, 
#         show_volume,
#         drop_rate,
#         adj_prc,
#         pred_move,
#         existing_dates,
#         mode,
#         dates_to_gen
#     )
    
#     if len(b) == 0:
#         empty_list.append(i)
#         print(i)
    
    

# 610
# 930
# 1858

    
    
# # Assuming `a[0][1]` is your DataFrame
# dftest = a[0][1]

# # Define the price-related columns that need to be checked
# price_cols = ['Close', 'Open', 'High', 'Low']

# # Step 1: Create a helper column to check if all price columns are `NaN`
# dftest['all_price_nan'] = dftest[price_cols].isna().all(axis=1)

# # Step 2: Identify groups of consecutive rows where `all_price_nan` is `True`
# dftest['group'] = (dftest['all_price_nan'] != dftest['all_price_nan'].shift()).cumsum()

# # Step 3: Calculate the size of each group
# group_sizes = dftest.groupby('group')['all_price_nan'].transform('size')

# # Step 4: Filter out rows where `all_price_nan` is `True` and the group size is greater than 1
# # Drop only if there are consecutive missing rows (group size > 1)
# df_filtered = dftest[~((dftest['all_price_nan']) & (group_sizes > 1))]

# # Step 5: Drop helper columns used for filtering
# df_filtered = df_filtered.drop(columns=['all_price_nan', 'group'])
    
    
# dftest = a[0][1]
# c = dftest[~dftest['Close'].ffill().isna()]
    
    
    