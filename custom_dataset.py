
from project_imports import *




class GraphDataset(Dataset):
    def __init__(self, path, transform=None, mode='train'):
        """
        Args:
            hdf5_path (str): Path to the HDF5 file containing the images and metadata.
            transform (callable, optional): Optional transform to be applied on a sample.

        Output:
            sample (dict): Dictionary containing the loaded image and the labels.
        """
        self.file_path = path
        self.transform = transform
        self.mode = mode
        self.dataset = None
        
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["labels"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve the image binary data from the HDF5 file
        binary_image = self.dataset['images'][idx]
    
        # Convert the binary data to a PIL image using BytesIO
        image = Image.open(io.BytesIO(binary_image)).convert('L')  # Convert to grayscale ('L' mode)

        if self.transform:
            image = self.transform(image)

        # Retrieve label and additional metadata
        label = self.dataset['labels'][idx]

        if self.mode == 'train':
            return {'image': image, 'ret20': label}

        elif self.mode == 'test':
            date = self.dataset['dates'][idx].decode('utf-8')
            permno = self.dataset['permnos'][idx].decode('utf-8')

            return {'image': image, 'ret20': label, 'date': date, 'permno': permno}

    def __del__(self):
        """Ensure the HDF5 file is closed when the object is deleted."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None




# class GraphDataset(Dataset):
#     def __init__(self, dataset_list, path, transform=None, mode='train'):
#         """
#         Args:
#             dataset_list (list of lists): List containing the file names of the images and their labels.
#             transform (callable, optional): Optional transform to be applied on a sample.

#         Output:
#             sample (dict): Dictionary containing the loaded image and the labels.
#         """
#         self.dataset_list = dataset_list
#         self.transform = transform
#         self.mode = mode
#         # self.active_table = active_table
#         # Set up a pooled engine
#         # self.engine = create_engine(POOL_CONNECTION_STRING, poolclass=QueuePool, max_overflow=10, pool_size=5)

#         # self.connection_string = CONNECTION_STRING
#         self.data_dir = f'{DATA_PATH}/{path}/'

#     def __len__(self):
#         return len(self.dataset_list)

#     def __getitem__(self, idx):
        
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # Extract the image filename from the dataset list
#         img = os.path.join(self.data_dir, self.dataset_list[idx][0])
#         image = Image.open(img).convert('L')  # 'L' mode means grayscale
        
#         if self.transform:
#             image = self.transform(image)
        
#         if self.mode == 'train':
            
#             label = self.dataset_list[idx][1]
            
#             return {'image': image, 'ret20': label}
        
#         elif self.mode == 'test':
            
#             label = self.dataset_list[idx][1]
#             date = self.dataset_list[idx][2]
#             permno = self.dataset_list[idx][3]
            
#             # Convert date to string if it's a pandas Timestamp
#             if isinstance(date, pd.Timestamp):
#                 date = date.strftime('%Y-%m-%d')
            
#             return {'image': image, 'ret20': label, 'date': date, 'permno': permno}
        
        
        
        
        
        

