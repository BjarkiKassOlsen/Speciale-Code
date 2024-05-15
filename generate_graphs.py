from project_imports import *


# Description: This file contains the functions to generate the graphs for each stock

def image_generator(permno, df, image_size, lookback, indicator, show_volume, pred_move, save_path, mode):
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

    # Remove rows with NA's
    df = df.dropna()

    dataset = []
    invalid_dates = []

    # # Connect to the database
    # connection = pyodbc.connect(CONNECTION_STRING)
    # cursor = connection.cursor()

    for day in range(len(df)-lookback):

        # Add the date to the image
        date_str = df.iloc[day+lookback]['Date'].strftime('%Y-%m-%d') # Formats the date as "YYYY-MM-DD"
        
        price_slice = df[day:day+lookback][['Open', 'High', 'Low', 'Close', 'MA']].reset_index(drop=True)
        volume_slice = df[day:day+lookback][['Volume']].reset_index(drop=True)

        # number of no transactions days > 0.2*look back days
        if (1.0*(price_slice[['Open', 'High', 'Low', 'Close']].sum(axis=1)/price_slice['Open'] == 4)).sum() > lookback//5: 
            print('No transactions')
            invalid_dates.append(df.iloc[day]['Date']) # trading dates not surviving the validation
            continue

        # project price into quantile
        price_slice = (price_slice - np.min(price_slice.values))/(np.max(price_slice.values) - np.min(price_slice.values))
        volume_slice = (volume_slice - np.min(volume_slice.values))/(np.max(volume_slice.values) - np.min(volume_slice.values))

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

            # Scale and shift the prices
            price_slice = price_slice.apply(lambda x: x*price_slice_range + price_slice_lower_limit).astype(int)
            
            # Scale and shift the volume
            volume_slice = volume_slice.apply(lambda x: x*(vol_space-1) + 1).astype(int) # Setting the lowest volume limit to 1

        # Create the image
        image = np.zeros(image_size)
        for i in range(len(price_slice)):
            # draw candlelist 
            image[price_slice.loc[i]['Open'], i*3] = 255
            image[price_slice.loc[i]['Low']:price_slice.loc[i]['High']+1, i*3+1] = 255
            image[price_slice.loc[i]['Close'], i*3+2] = 255
            # draw indicators
            for ind in ['MA']:
                image[price_slice.loc[i][ind], i*3:i*3+2] = 255
            # draw volume bars
            if show_volume:
                image[:volume_slice.loc[i]['Volume'], i*3+1] = 255


        # Flip the y-axis of the image matrix, so we have a normal graph with volume at the bottom
        flipped_image = np.flipud(image)

        # Add the date to the image
        # date_str = df.iloc[day+lookback]['Date'].strftime('%Y-%m-%d') # Formats the date as "YYYY-MM-DD"

        # # Save the image
        # if show_volume:
        #     image_name = f'I{lookback}VolTInd{list(indicator[0].values())[0]}/{permno}_{date_str}.png'
        # else:
        #     image_name = f'I{lookback}VolFInd{list(indicator[0].values())[0]}/{permno}_{date_str}.png'

        # image_name = f'{save_path}/{permno}_{date_str}.png'
        image_name = f'{permno}_{date_str}.png'

        # Ensure the image is in 8-bit unsigned integer format
        image_8bit = np.clip(flipped_image, 0, 255).astype(np.uint8)

        # Convert the matrix to a PIL image in 'L' mode (grayscale)
        image_PIL = Image.fromarray(image_8bit, 'L')

        # Save the image as a PNG file
        image_PIL.save(os.path.join(DATA_PATH, f'{save_path}/{image_name}'))

        # buffer = io.BytesIO()
        # image_PIL.save(buffer, format='PNG')
        # binary_image = buffer.getvalue()

        # Insert the image into the database

        # # Check if the image already exists
        # cursor.execute(f"SELECT COUNT(*) FROM {active_table} WHERE FileName = ?", (image_name,))
        # exists = cursor.fetchone()[0] > 0

        # if exists:
        #     # Update existing record
        #     try:
        #         cursor.execute(f"UPDATE {active_table} SET FileData = ? WHERE FileName = ?", (binary_image, image_name))
        #         connection.commit()
        #     except Exception as e:
        #         print(f"Failed to update image in the database: {e}")
        # else:
        #     # Insert new record
        #     try:
        #         cursor.execute(f"INSERT INTO {active_table} (FileName, FileData) VALUES (?, ?)", (image_name, binary_image))
        #         connection.commit()
        #     except Exception as e:
        #         print(f"Failed to insert new image into the database: {e}")

        # try:
        #     cursor.execute(f"INSERT INTO {active_table} (FileName, FileData) VALUES (?, ?)", (image_name, binary_image))
        #     connection.commit()
        # except Exception as e:
        #     print(f"Failed to insert image into the database: {e}")

        # os.path.join(directory, image_name)

        if pred_move:
            # Add the actual return label to the image
            label_ret5 = df.iloc[day+lookback]['Ret5'].round(3)
            label_ret20 = df.iloc[day+lookback]['Ret20'].round(3)
            label_ret60 = df.iloc[day+lookback]['Ret60'].round(3)

            entry = [image_name, label_ret5, label_ret20, label_ret60]
            dataset.append(entry)
        else:
            # Add the return direction label to the image, 1 if positive, 0 if negative
            label_ret5 = 1 if df.iloc[day+lookback]['Ret5'] > 0 else 0
            label_ret20 = 1 if df.iloc[day+lookback]['Ret20'] > 0 else 0
            label_ret60 = 1 if df.iloc[day+lookback]['Ret60'] > 0 else 0

            entry = [image_name, label_ret5, label_ret20, label_ret60]
            dataset.append(entry)

    # connection.close()

    return dataset#, invalid_dates

# def image_generator(permno, df, image_size, lookback, indicator, show_volume, pred_move, h5file, mode):
#     """
#     Generate the graphs for each stock

#     input:
#         df: dataframe with the stock data
#         image_size: size of the image to output
#         start_date: start date of the image to generate
#         end_date: end date of the image to generate
#         indicators: list of indicators to plot, i.e. ['MA': 20]
#         show_volume: boolean to show the volume in the graph
#         mode: 'train' or 'test'

#     output:

#     """
    

#     if len(indicator) > 0:
#         for ind in indicator:
#             for key, value in ind.items():
#                 df = df.assign(**{key: df['Close'].rolling(window=value).mean()})

#     # Remove rows with NA's
#     df = df.dropna()

#     dataset = []
#     invalid_dates = []

#     image_index = 0 

#     with h5py.File(h5file, 'a') as h5f:
#         for day in range(len(df)-lookback):

#             # Add the date to the image
#             date_str = df.iloc[day+lookback]['Date'].strftime('%Y-%m-%d') # Formats the date as "YYYY-MM-DD"
            
#             price_slice = df[day:day+lookback][['Open', 'High', 'Low', 'Close', 'MA']].reset_index(drop=True)
#             volume_slice = df[day:day+lookback][['Volume']].reset_index(drop=True)

#             # number of no transactions days > 0.2*look back days
#             if (1.0*(price_slice[['Open', 'High', 'Low', 'Close']].sum(axis=1)/price_slice['Open'] == 4)).sum() > lookback//5: 
#                 print('No transactions')
#                 invalid_dates.append(df.iloc[day]['Date']) # trading dates not surviving the validation
#                 continue

#             # project price into quantile
#             price_slice = (price_slice - np.min(price_slice.values))/(np.max(price_slice.values) - np.min(price_slice.values))
#             volume_slice = (volume_slice - np.min(volume_slice.values))/(np.max(volume_slice.values) - np.min(volume_slice.values))

#             # project price and volume into pixel
#             if not show_volume:
#                 price_slice = price_slice.apply(lambda x: x*(image_size[0]-2) + 1).astype(int)
#             else:
#                 # Define the volume allocation space on the graph
#                 vol_space = int(0.2*(image_size[0]))

#                 # Define the range of the price_slice
#                 price_slice_lower_limit = vol_space + 1
#                 price_slice_upper_limit = image_size[0] - 1
#                 price_slice_range = price_slice_upper_limit - price_slice_lower_limit

#                 # Scale and shift the prices
#                 price_slice = price_slice.apply(lambda x: x*price_slice_range + price_slice_lower_limit).astype(int)
                
#                 # Scale and shift the volume
#                 volume_slice = volume_slice.apply(lambda x: x*(vol_space-1) + 1).astype(int) # Setting the lowest volume limit to 1

#             # Create the image
#             image = np.zeros(image_size)
#             for i in range(len(price_slice)):
#                 # draw candlelist 
#                 image[price_slice.loc[i]['Open'], i*3] = 255
#                 image[price_slice.loc[i]['Low']:price_slice.loc[i]['High']+1, i*3+1] = 255
#                 image[price_slice.loc[i]['Close'], i*3+2] = 255
#                 # draw indicators
#                 for ind in ['MA']:
#                     image[price_slice.loc[i][ind], i*3:i*3+2] = 255
#                 # draw volume bars
#                 if show_volume:
#                     image[:volume_slice.loc[i]['Volume'], i*3+1] = 255


#             # Flip the y-axis of the image matrix, so we have a normal graph with volume at the bottom
#             flipped_image = np.flipud(image)

#             # image_name = f'{save_path}/{permno}_{date_str}.png'
#             image_name = f'{permno}_{date_str}.png'

#             # Ensure the image is in 8-bit unsigned integer format
#             image_8bit = np.clip(flipped_image, 0, 255).astype(np.uint8)

#             # # Convert the matrix to a PIL image in 'L' mode (grayscale)
#             # image_PIL = Image.fromarray(image_8bit, 'L')

#             # # Save the image as a PNG file
#             # image_PIL.save(os.path.join(DATA_PATH, image_name))

#             # os.path.join(directory, image_name)

#             if pred_move:
#                 # Add the actual return label to the image
#                 label_ret5 = df.iloc[day+lookback]['Ret5'].round(3)
#                 label_ret20 = df.iloc[day+lookback]['Ret20'].round(3)
#                 label_ret60 = df.iloc[day+lookback]['Ret60'].round(3)

#                 entry = [image_name, label_ret5, label_ret20, label_ret60]
#                 dataset.append(entry)
#             else:
#                 # Add the return direction label to the image, 1 if positive, 0 if negative
#                 label_ret5 = 1 if df.iloc[day+lookback]['Ret5'] > 0 else 0
#                 label_ret20 = 1 if df.iloc[day+lookback]['Ret20'] > 0 else 0
#                 label_ret60 = 1 if df.iloc[day+lookback]['Ret60'] > 0 else 0

#                 entry = [image_name, label_ret5, label_ret20, label_ret60]
#                 dataset.append(entry)

#             # Write directly to the HDF5 dataset
#             h5f['images'][image_index] = image_8bit
#             # Assuming labels are also prepared
#             h5f['labels'][image_index] = [label_ret5, label_ret20, label_ret60]
#             image_index += 1


#     return dataset#, invalid_dates


def show_single_graph(entry, path):
    
    # # Connect to the database
    # connection = pyodbc.connect(CONNECTION_STRING)
    # cursor = connection.cursor()

    # Load the image from file
    image_path = f'{DATA_PATH}/{path}/{entry[0]}'
    # image_path = f'{entry[0]}'
    image = Image.open(image_path)

    # # Load the image from the database
    # cursor.execute(f"SELECT FileData FROM {active_table} WHERE FileName = ?", (entry[0],))
    # row = cursor.fetchone()

    # connection.close()
    
    # if row is not None:
    #     image_data = row[0]  # Get the image data from the result set

    #     # Convert the binary data to a format that can be read by Image.open
    #     image_stream = io.BytesIO(image_data)
        
    #     # Open the image using PIL
    #     image = Image.open(image_stream)
        
    # else:
    #     print("No image found for the given filename.")

    # Split the string and extract the date
    date_part = entry[0].split('_')[1].split('.')[0]
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    plt.figure()
    plt.imshow(image_array, cmap=plt.get_cmap('gray'))
    plt.title(f'Ret5: {entry[1]}, Ret20: {entry[2]}, Ret60: {entry[3]}\n Last Date is: {date_part}')
    plt.axis('off')  # Optional: Hide the axes
    plt.tight_layout()
    plt.show()

# def show_single_graph(entry, hdf5_path):
    
#     # Load the HDF5 file
#     with h5py.File(hdf5_path, 'r') as h5f:
#         # Extract the image index from the filename
#         image_index = entry[0].split('_')[0]
        
#         # Load the image data from HDF5
#         image_data = h5f['images'][int(image_index)]
        
#         # Convert the numpy array back to an image
#         image = Image.fromarray(image_data, 'L')
        
#     # Split the string and extract the date
#     date_part = entry[0].split('_')[1].split('.')[0]
    
#     # Convert image to numpy array
#     image_array = np.array(image)
    
#     plt.figure()
#     plt.imshow(image_array, cmap=plt.get_cmap('gray'))
#     plt.title(f'Ret5: {entry[1]}, Ret20: {entry[2]}, Ret60: {entry[3]}\n Last Date is: {date_part}')
#     plt.axis('off')  # Optional: Hide the axes
#     plt.tight_layout()
#     plt.show()



class GraphDataset():
    def __init__(self, df, win_size, mode, label, indicator = [], show_volume=False, predict_movement=False, parallel_num=-1):

        assert win_size in [5, 20, 60], f'Wrong look back days: {win_size}'
        assert mode in ['train', 'test', 'inference'], f'Type Error: {mode}'
        assert label in ['Ret5', 'Ret20', 'Ret60'], f'Wrong Label: {label}'

        self.df = df
        self.mode = mode
        self.label = label
        self.indicator = indicator
        self.show_volume = show_volume
        self.pred_move = predict_movement
        self.parallel_num = parallel_num
        

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
        
        # # Connect to the database
        # conn = pyodbc.connect(CONNECTION_STRING)
        # cursor = conn.cursor()

        # Save the images
        if self.show_volume:
            table_name = f'I{self.window_size}VolTInd{list(self.indicator[0].values())[0]}'
        else:
            table_name = f'I{self.window_size}VolFInd{list(self.indicator[0].values())[0]}'

        # # Check if the table exists
        # cursor.execute(f"SELECT OBJECT_ID('{table_name}')")
        # result = cursor.fetchone()

        # if result[0] is None:
        #     # Table doesn't exist, create it
        #     cursor.execute(f"CREATE TABLE {table_name} (FileName NVARCHAR(255) PRIMARY KEY, FileData VARBINARY(MAX))")
        #     conn.commit()
        #     print(f"Table: '{table_name}' created successfully.")
        # else:
        #     print(f"Table: '{table_name}' already exists.")

        # conn.close()

        # Check if the directory exists, if not, create it
        if not os.path.exists(table_name):
            os.makedirs(table_name)

        # Utilize parallel processing to generate the images for all the stocks quickly
        dataset_all = Parallel(n_jobs=self.parallel_num)(delayed(image_generator)(
                                        g[0], g[1], image_size = self.image_size, 
                                        lookback = self.window_size,
                                        indicator = self.indicator, 
                                        show_volume = self.show_volume,
                                        pred_move = self.pred_move,
                                        save_path = f'{DATA_PATH}/{table_name}/',
                                        mode = self.mode
                                        ) for g in tqdm(self.df.groupby('permno'), 
                                                        desc=f'Generating Images'))

        # Concatenate the dataset
        image_set = []
        for data in dataset_all:
            for row in data:
                image_set.append(row)
        dataset_all = []

        # Convert image_set to a DataFrame
        image_set = pd.DataFrame(image_set)

        image_set.to_csv(f'{DATA_PATH}/{table_name}_dataset.csv', index=False)

        return image_set, table_name

    # def generate_images(self):

    #     # Save the images
    #     if self.show_volume:
    #         table_name = f'I{self.window_size}VolTInd{list(self.indicator[0].values())[0]}'
    #     else:
    #         table_name = f'I{self.window_size}VolFInd{list(self.indicator[0].values())[0]}'

    #     hdf5_path = f'{DATA_PATH}/{table_name}_dataset.h5'
    #     with h5py.File(hdf5_path, 'w') as h5f:
    #         # Prepare datasets for images and labels
    #         # Assuming images are grayscale and of size 256x256
    #         images_dset = h5f.create_dataset('images', (4000000, 256, 256), dtype='uint8', compression='gzip')
    #         labels_dset = h5f.create_dataset('labels', (4000000, 3), dtype='float32', compression='gzip')

    #         # Utilize parallel processing to generate the images for all stocks quickly
    #         results = Parallel(n_jobs=self.parallel_num)(delayed(image_generator)(
    #             g[0], g[1], image_size=self.image_size,
    #             lookback=self.window_size, indicator=self.indicator,
    #             show_volume=self.show_volume, pred_move=self.pred_move,
    #             h5file=hdf5_path, dataset_images=images_dset, dataset_labels=labels_dset
    #         ) for g in tqdm(self.df.groupby('permno'), desc='Generating Images'))

    #     print("All data has been saved to HDF5.")
    #     return hdf5_path

    # def generate_images(self):

    #     # Save the images
    #     if self.show_volume:
    #         table_name = f'I{self.window_size}VolTInd{list(self.indicator[0].values())[0]}'
    #     else:
    #         table_name = f'I{self.window_size}VolFInd{list(self.indicator[0].values())[0]}'

    #     # Utilize parallel processing to generate the images for all the stocks quickly
    #     dataset_all = Parallel(n_jobs=self.parallel_num)(delayed(image_generator)(
    #                                     g[0], g[1], image_size=self.image_size, 
    #                                     lookback=self.window_size,
    #                                     indicator=self.indicator, 
    #                                     show_volume=self.show_volume,
    #                                     pred_move=self.pred_move,
    #                                     mode=self.mode
    #                                     ) for g in tqdm(self.df.groupby('permno'), 
    #                                                     desc=f'Generating Images'))

    #     # Concatenate the dataset
    #     image_set = []
    #     for data in dataset_all:
    #         for row in data:
    #             image_set.append(row)
    #     dataset_all = []

    #     # Convert image_set to a DataFrame
    #     image_set = pd.DataFrame(image_set, columns=['Image', 'Ret5', 'Ret20', 'Ret60'])

    #     # Save the images and labels to HDF5
    #     with h5py.File(f'{DATA_PATH}/{table_name}_dataset.h5', 'w') as h5f:
    #         # Create a dataset for images
    #         images_shape = (len(image_set), *self.image_size)
    #         images = h5f.create_dataset('images', shape=images_shape, dtype=np.uint8, compression='gzip')

    #         # # Save images
    #         # for i, img_data in enumerate(image_set['Image']):
    #         #     buffer = io.BytesIO(img_data)
    #         #     image_PIL = Image.open(buffer)
    #         #     image_array = np.array(image_PIL)
    #         #     images[i] = image_array

    #         # Save images with a progress bar
    #         for i, img_data in tqdm(enumerate(image_set['Image']), total=len(image_set), desc='Saving Images to HDF5'):
    #             buffer = io.BytesIO(img_data)
    #             image_PIL = Image.open(buffer)
    #             image_array = np.array(image_PIL)
    #             images[i] = image_array

    #         # Save labels
    #         h5f.create_dataset('ret5', data=image_set['Ret5'].values, compression='gzip')
    #         h5f.create_dataset('ret20', data=image_set['Ret20'].values, compression='gzip')
    #         h5f.create_dataset('ret60', data=image_set['Ret60'].values, compression='gzip')

    #     return image_set, table_name

        

    # def generate_images(self):
    #     # Define a helper function inside the method to handle the execution and pairing with permno
    #     def process_group(g):
    #         permno, df = g
    #         return permno, image_generator(
    #             df, 
    #             image_size=self.image_size, 
    #             lookback=self.window_size,
    #             indicator=self.indicator, 
    #             show_volume=self.show_volume,
    #             mode=self.mode
    #         )

    #     # Utilize parallel processing to generate the images for all the stocks quickly
    #     dataset_all = Parallel(n_jobs=self.parallel_num)(
    #         delayed(process_group)(g) for g in tqdm(self.df.groupby('permno'), desc='Generating Images')
    #     )

    #     return dataset_all

    


