
from project_imports import *


from PIL import UnidentifiedImageError
# import imghdr

class GraphDataset(Dataset):
    def __init__(self, path, transform=None, mode='train', model = 'CNN'):#, indices = None, data_len = None):
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
        self.model = model
        # self.indices = indices
        # self.dataset_len = data_len
        self.dataset = None
        
        with h5py.File(self.file_path, 'r') as file:
            # Use the date indices in the dataset to filter the data
            # self.indices = [i for i, date in enumerate(file["dates"]) if start <= int(date.decode('utf-8')) < end]
            # self.dataset_len = len(self.indices)
            
            # date = file['dates']
            self.dataset_len = len(file['dates'])
            

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # if self.indices:
        #     # Retrieve the actual index in the HDF5 file
        #     idx = self.indices[idx]
            
        # Retrieve label
        label = self.dataset['labels'][idx]

        if self.model == 'CNN':

            # Retrieve the image binary data from the HDF5 file
            binary_image = self.dataset['images'][idx]
        
            # # Convert the binary data to a PIL image using BytesIO
            # image = Image.open(io.BytesIO(binary_image)).convert('L')  # Convert to grayscale ('L' mode)
            
            try:
                # Attempt to open the image
                image = Image.open(io.BytesIO(binary_image)).convert('L')  # Convert to grayscale ('L' mode)
            except UnidentifiedImageError as e:
                print(f"UnidentifiedImageError: Skipping index {idx}, unable to open image.")
                return self.__getitem__(idx + 1)  # Skip to the next item or return a placeholder
    
            
            if self.transform:
                image = self.transform(image)
    
            if self.mode == 'train':
                return {'image': image, 'label': label}
    
            elif self.mode == 'test':
                date = self.dataset['dates'][idx].decode('utf-8')
                permno = self.dataset['permnos'][idx].decode('utf-8')
                me = self.dataset['ME'][idx]
    
                return {'image': image, 'label': label, 'yyyymm': date, 'permno': permno, 'ME': me}
            
        elif self.model == 'XGBoost':
            
            # Retrieve the characterics from the hdf5 file
            chars = self.dataset['chars'][idx]
            
            if self.transform:
                image = self.transform(image)
                
            if self.mode == 'train':
                return {'chars': chars, 'label': label}
    
            elif self.mode == 'test':
                date = self.dataset['dates'][idx].decode('utf-8')
                permno = self.dataset['permnos'][idx].decode('utf-8')
                me = self.dataset['ME'][idx]
    
                return {'image': image, 'label': label, 'yyyymm': date, 'permno': permno, 'ME': me}
    

    def __del__(self):
        """Ensure the HDF5 file is closed when the object is deleted."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None



        
        
        
        
class FinancialDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df
        self.X = torch.tensor(df[['XGB_prob', 'CNN_prob']].values, dtype=torch.float32)  # Pre-convert to tensors
        self.y = torch.tensor(df['label'].values, dtype=torch.float32)  # Pre-convert to tensors
        
        self.permno = df['permno'].values  # Keep other columns for test mode
        self.date = df['date'].values
        self.me = df['ME'].values
        self.mode = mode  # 'train' or 'test'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        
        if self.mode == 'train':
            # In training mode, return only X and y
            return X, y
        
        elif self.mode == 'test':
            # In test mode, return additional information (e.g., permno, date)
            permno = self.permno[idx]
            date = self.date[idx]
            me = self.me[idx]
            return {
                'X': X,
                'label': y,
                'permno': permno,
                'date': date,
                'ME': me
            }

        

