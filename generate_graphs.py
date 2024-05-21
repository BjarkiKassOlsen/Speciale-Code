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

        # image_name = f'{save_path}/{permno}_{date_str}.png'
        image_name = f'{permno}_{date_str}.png'

        # Ensure the image is in 8-bit unsigned integer format
        image_8bit = np.clip(flipped_image, 0, 255).astype(np.uint8)

        # Convert the matrix to a PIL image in 'L' mode (grayscale)
        image_PIL = Image.fromarray(image_8bit, 'L')

        # Save the image as a PNG file
        image_PIL.save(os.path.join(DATA_PATH, f'{save_path}/{image_name}'))


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

    return dataset



def show_single_graph(entry, path):

    # Load the image from file
    image_path = f'{DATA_PATH}/{path}/{entry[0]}'
    # image_path = f'{entry[0]}'
    image = Image.open(image_path)

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

        # Save the images
        if self.show_volume:
            table_name = f'I{self.window_size}VolTInd{list(self.indicator[0].values())[0]}'
        else:
            table_name = f'I{self.window_size}VolFInd{list(self.indicator[0].values())[0]}'

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
                                        save_path = table_name,
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

    


