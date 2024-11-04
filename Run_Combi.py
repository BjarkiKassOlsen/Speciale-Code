
from project_imports import *

# from importlib import reload
import custom_dataset


run = neptune.init_run(
    project="bjarki/Speciale",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzc0ZjBjMy0wYzBjLTQwMGYtODExYS1iNDM1MjAxZDdlNWMifQ==",
)  # your credentials



# Define the model
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         # A single linear layer that takes 2 inputs and produces 1 output
#         self.fc = nn.Linear(2, 2)
    
#     def forward(self, x):
#         # Forward pass: input -> linear layer -> sigmoid for probability output
#         x = self.fc(x)
#         # x = torch.sigmoid(x)  # Apply sigmoid to get a probability output
#         return x

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # A single linear layer that takes 2 inputs and produces 2 outputs
        self.fc1 = nn.Linear(2, 6)  # First fully connected layer
        self.fc2 = nn.Linear(6, 2)  # Second fully connected layer
        self.relu = nn.ReLU()       # ReLU activation
        self.dropout = nn.Dropout(p=0.75)  # Dropout layer
        self.bn1 = nn.BatchNorm1d(6)  # Batch normalization for first layer

    def forward(self, x):
        x = self.fc1(x)
        
        # Only apply batch normalization if batch size is greater than 1
        if x.size(0) > 1:
            x = self.bn1(x)

        x = self.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Second fully connected layer
        return x
    


# Function to train the SimpleNN model with early stopping and GPU support
def train_simple_nn(device, train_loader, valid_loader, patience=3):
    
    model = SimpleNN().to(device)  # Move the model to GPU (if available)
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100  # Max epochs
    best_valid_loss = float('inf')  # Initialize best validation loss to infinity
    epochs_without_improvement = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_train_samples = 0  # To calculate average loss
        
        # Training loop with tqdm progress
        for X_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            # Move data to GPU (if available)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            # Convert y_batch to 0's and 1's (binary target)
            y_batch = (y_batch > 0.0).long()  # CrossEntropyLoss expects class labels as long
            
            # Calculate loss (raw outputs are logits, no need to apply sigmoid or softmax)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)  # Sum up the loss weighted by batch size
            num_train_samples += X_batch.size(0)  # Keep track of the number of training samples

        # Validation loop
        valid_loss = 0.0
        num_valid_samples = 0  # To calculate average validation loss
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                
                # Move data to GPU (if available)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                
                # Convert y_batch to 0's and 1's
                y_batch = (y_batch > 0.0).long()
                
                # Calculate validation loss
                loss = criterion(outputs, y_batch)
                
                valid_loss += loss.item() * X_batch.size(0)  # Sum up the loss weighted by batch size
                num_valid_samples += X_batch.size(0)  # Keep track of the number of validation samples

        # Calculate average train and validation loss
        avg_train_loss = train_loss / num_train_samples
        avg_valid_loss = valid_loss / num_valid_samples

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')
        
        # Early stopping: check if validation loss improved
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss  # Update best validation loss
            epochs_without_improvement = 0  # Reset counter if improved
        else:
            epochs_without_improvement += 1  # Increment counter if no improvement
        
        # Check if early stopping condition is met
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return model





data = pd.read_csv(f'{DATA_PATH}/Combi_Model/data.csv')


train_start = 199301
train_window_years = 7
valid_window_years = 2
test_window = 1
# train_ratio = 0.8



results = []  # List to store DataFrame results

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
device = 'cpu'

while train_start + train_window_years * 100 < 202201:
    # Define train, validation, and test periods
    train_end = train_start + train_window_years * 100
    valid_start = train_end - valid_window_years * 100
    test_end = train_end + test_window
    
    print(f"Training from {train_start} to {train_end}, Testing from {train_end} to {test_end}")
    
    # Split the dataset
    train_data = data[(data['date'] >= train_start) & (data['date'] < valid_start)].copy()
    valid_data = data[(data['date'] >= valid_start) & (data['date'] < train_end)]
    test_data = data[(data['date'] >= train_end) & (data['date'] < test_end)]
    
    # Initialize DataLoader for each split
    train_loader = DataLoader(custom_dataset.FinancialDataset(train_data, mode='train'), batch_size=32, shuffle=False, pin_memory=True, num_workers=0)
    valid_loader = DataLoader(custom_dataset.FinancialDataset(valid_data, mode='train'), batch_size=32, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(custom_dataset.FinancialDataset(test_data, mode='test'), batch_size=32, shuffle=False, pin_memory=True, num_workers=0)
    
    # Train the SimpleNN model
    trained_model = train_simple_nn(device, train_loader, valid_loader, patience=3)
    
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Wrap your data loader with tqdm for a progress bar
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Predicting'):
            outputs = trained_model(batch['X'].to(device))

            # Record logits along with permno and date for each item in the batch
            batch_results = pd.DataFrame({
                'permno': batch['permno'],
                'date': batch['date'],
                'label': batch['label'].cpu().numpy(),
                'ME': batch['ME'].cpu().numpy(),
                'neg_ret': outputs[:, 0].cpu().numpy(),
                'pos_ret': outputs[:, 1].cpu().numpy()
            })

            results.append(batch_results)
    
    # Move the window one month forward
    if train_start % 100 != 12:
        train_start += test_window
    else:
        train_start += 100 - 11


results = pd.concat(results, ignore_index=True)


results.to_csv(f'{DATA_PATH}/returns/Combi_Model/Preds.csv')


run.stop()