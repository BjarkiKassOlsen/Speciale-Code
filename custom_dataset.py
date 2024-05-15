
from project_imports import *


class GraphDataset(Dataset):
    def __init__(self, dataset_list, path, transform=None):
        """
        Args:
            dataset_list (list of lists): List containing the file names of the images and their labels.
            transform (callable, optional): Optional transform to be applied on a sample.

        Output:
            sample (dict): Dictionary containing the loaded image and the labels.
        """
        self.dataset_list = dataset_list
        self.transform = transform
        # self.active_table = active_table
        # Set up a pooled engine
        # self.engine = create_engine(POOL_CONNECTION_STRING, poolclass=QueuePool, max_overflow=10, pool_size=5)

        # self.connection_string = CONNECTION_STRING
        self.data_dir = f'{DATA_PATH}/{path}/'

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the image filename from the dataset list
        img = os.path.join(self.data_dir, self.dataset_list[idx][0])
        image = Image.open(img).convert('L')  # 'L' mode means grayscale
        labels = self.dataset_list[idx][1:]

        # # Extract the image filename from the dataset list
        # image_filename = self.dataset_list[idx][0]
        # labels = self.dataset_list[idx][1:]

        # # Connect to the database
        # conn = pyodbc.connect(self.connection_string)
        # cursor = conn.cursor()

        # try:
        #     # Retrieve the image data from the database
        #     cursor.execute(f"SELECT FileData FROM {self.active_table} WHERE FileName = ?", (image_filename,))
        #     image_data = cursor.fetchone()[0]
        #     image_stream = io.BytesIO(image_data)

        #     # Load image from binary stream
        #     image = Image.open(image_stream).convert('L')  # Convert to grayscale if needed

        # finally:
        #     conn.close()


        # # Retrieve the image data from the database
        # self.cursor.execute(f"SELECT FileData FROM {self.active_table} WHERE FileName = ?", (image_filename,))
        # image_data = self.cursor.fetchone()[0]
        # image_stream = io.BytesIO(image_data)

        # # Load image from binary stream
        # image = Image.open(image_stream).convert('L')  # Convert to grayscale if needed

        # with self.engine.connect() as conn:
        #     result = conn.execute(f"SELECT FileData FROM {self.active_table} WHERE FileName = ?", (image_filename,))
        #     image_data = result.fetchone()[0]

        # image_stream = io.BytesIO(image_data)
        # image = Image.open(image_stream).convert('L')

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'ret5': labels[0], 'ret20': labels[1], 'ret60': labels[2]}
        
        # # Split the string and extract the date
        # date_part = self.dataset_list[idx][0].split('_')[1].split('.')[0]

        return sample#, date_part
    

# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import io
# from sqlalchemy import create_engine, text
# from sqlalchemy.pool import QueuePool

# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import io
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import scoped_session, sessionmaker
# from sqlalchemy.pool import QueuePool

# class GraphDataset(Dataset):
#     def __init__(self, dataset_list, active_table, transform=None):
#         """
#         Args:
#             dataset_list (list of lists): List containing the file names of the images and their labels.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.dataset_list = dataset_list
#         self.transform = transform
#         self.active_table = active_table
#         self.connection_string = POOL_CONNECTION_STRING

#         # Initialize the connection pool
#         self.engine = create_engine(self.connection_string, poolclass=QueuePool, pool_size=10, max_overflow=20)
#         self.Session = scoped_session(sessionmaker(bind=self.engine))

#     def __len__(self):
#         return len(self.dataset_list)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # Extract the image filename from the dataset list
#         image_filename = self.dataset_list[idx][0]
#         labels = self.dataset_list[idx][1:]

#         # Retrieve the image data from the database
#         session = self.Session()
#         try:
#             result = session.execute(text(f"SELECT FileData FROM {self.active_table} WHERE FileName = :filename"),
#                                      {"filename": image_filename})
#             image_data = result.fetchone()[0]
#             image_stream = io.BytesIO(image_data)

#             # Load image from binary stream
#             image = Image.open(image_stream).convert('L')  # Convert to grayscale if needed
#         finally:
#             session.close()

#         if self.transform:
#             image = self.transform(image)

#         sample = {'image': image, 'ret5': labels[0], 'ret20': labels[1], 'ret60': labels[2]}

#         return sample
