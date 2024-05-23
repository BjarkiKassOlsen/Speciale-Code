from project_imports import *

import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define helper functions
# @staticmethod
def _update_running_metrics(loss, labels, preds, running_metrics):
    running_metrics["running_loss"] += loss.item() * len(labels)
    running_metrics["running_correct"] += (preds == labels).sum().item()
    running_metrics["TP"] += (preds * labels).sum().item()
    running_metrics["TN"] += ((preds - 1) * (labels - 1)).sum().item()
    running_metrics["FP"] += (preds * (labels - 1)).sum().abs().item()
    running_metrics["FN"] += ((preds - 1) * labels).sum().abs().item()

# @staticmethod
def _generate_epoch_stat(epoch, learning_rate, num_samples, running_metrics):
    TP, TN, FP, FN = (running_metrics["TP"],
                      running_metrics["TN"],
                      running_metrics["FP"],
                      running_metrics["FN"])
    
    epoch_stat = {"epoch": epoch, "lr": "{:.2E}".format(learning_rate)}
    epoch_stat["diff"] = 1.0 * ((TP + FP) - (TN + FN)) / num_samples
    epoch_stat["loss"] = running_metrics["running_loss"] / num_samples
    epoch_stat["accy"] = 1.0 * running_metrics["running_correct"] / num_samples
    epoch_stat["MCC"] = (
        np.nan
        if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0
        else (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    )
    return epoch_stat



def evaluate(model, dataloaders_dict, pred_win, criterion, new_label=None):
    
    print("Evaluating model")
    model.to(device)
    res_dict = {}
    for subset in dataloaders_dict.keys():
        
        data_iterator = tqdm(dataloaders_dict[subset], leave=True, unit="batch")
        data_iterator.set_description("Evaluation: ")
        
        model.eval()
        running_metrics = {
            "running_loss": 0.0,
            "running_correct": 0.0,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }
        for batch in data_iterator:
            inputs = batch['image'].to(device, dtype=torch.float32)
            
            if new_label is not None:
                labels = (
                    torch.Tensor([new_label])
                    .repeat(inputs.shape[0])
                    .to(device, dtype=torch.float)
                )
            else:
                labels = batch[f'ret{pred_win}'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            _update_running_metrics(loss, labels, preds, running_metrics)
            # del inputs, labels, data, target
        num_samples = len(dataloaders_dict[subset].dataset)
        epoch_stat = _generate_epoch_stat(-1, -1, num_samples, running_metrics)
        
        data_iterator.set_postfix(epoch_stat)
        data_iterator.update()
        print(epoch_stat)
        res_dict[subset] = {
            metric: epoch_stat[metric] for metric in ["loss", "accy", "MCC", "diff"]
        }
    del model
    torch.cuda.empty_cache()
    return res_dict


def train_n_epochs(n_epochs, model, pred_win, train_loader, valid_loader, 
                   early_stop, early_stop_patience, lr=1e-4, regression_label=False, run=None):

    # Define a loss function and optimizer
    if regression_label:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define a dictionary to store the dataloaders
    dataloaders_dict = {"train": train_loader, "valid": valid_loader}

    # Keep track of the best model
    best_validate_metrics = {"loss": 10.0, "accy": 0.0, "MCC": 0.0, "epoch": 0}
    best_model = copy.deepcopy(model.state_dict())
    train_metrics = {"prev_loss": 10.0, "pattern_accy": -1}

    # Store stats for each epoch
    epoch_stats_history = {'train': [], 'valid': []}

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
            
            for batch in data_iterator:
                
                inputs = batch['image'].to(device, dtype=torch.float32)
                labels = batch[f'ret{pred_win}'].to(device, dtype=torch.float32)

                with torch.set_grad_enabled(phase == "train"):
                    
                    # Make predictions
                    outputs = model(inputs)

                    # Convert labels to 1's and 0's if regression_label is False
                    if not regression_label:
                        labels = torch.where(labels > 0, torch.tensor(1, device=labels.device), torch.tensor(0, device=labels.device))
                    
                    # Compute loss
                    loss = criterion(outputs, labels) 
                    
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if regression_label:
                    # Convert the predictions to class labels
                    preds = torch.where(outputs > 0, torch.tensor(1, device=outputs.device), torch.tensor(0, device=outputs.device)).squeeze(1)
                    labels = torch.where(labels > 0, torch.tensor(1, device=labels.device), torch.tensor(0, device=labels.device))
                
                else:
                    # Convert the predictions to class labels
                    preds = torch.max(outputs, 1)[1]

                # Update the running metrics
                _update_running_metrics(loss, labels, preds, running_metrics)

                # # Delete the variables to free up memory
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

        # Save the epoch stats history
        epoch_stats_history["train"].append(epoch_stat_train)
        epoch_stats_history["valid"].append(epoch_stat_valid)

        if run is not None:

            # Upload stats to neptne
            run["train/loss"].append(epoch_stat_train["loss"])
            run["train/accuracy"].append(epoch_stat_train["accy"])
            run["train/MCC"].append(epoch_stat_train["MCC"])
            run["train/diff"].append(epoch_stat_train["diff"])

            run["valid/loss"].append(epoch_stat_valid["loss"])
            run["valid/accuracy"].append(epoch_stat_valid["accy"])
            run["valid/MCC"].append(epoch_stat_valid["MCC"])
            run["valid/diff"].append(epoch_stat_valid["diff"])
        
        print(f'Current epoch: {epoch}. \nBest epoch: {best_validate_metrics["epoch"]}')

        if early_stop and (epoch - best_validate_metrics["epoch"]) >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break


    # Load the best model weights
    model.load_state_dict(best_model)
    best_validate_metrics["model_state_dict"] = model.state_dict().copy()

    # train_metrics = evaluate(model, {"train": dataloaders_dict["train"]}, 
    #                          pred_win, criterion)["train"]
    # train_metrics["epoch"] = best_validate_metrics["epoch"]

    del best_validate_metrics["model_state_dict"]

    return epoch_stats_history, best_validate_metrics, model



# Plot the stats for each epoch
def plot_epoch_stats(epoch_stats_dict, run=None):

    epochs = [stat['epoch'] for stat in epoch_stats_dict['train']]

    train_loss = [stat['loss'] for stat in epoch_stats_dict['train']]
    valid_loss = [stat['loss'] for stat in epoch_stats_dict['valid']]

    train_accy = [stat['accy'] for stat in epoch_stats_dict['train']]
    valid_accy = [stat['accy'] for stat in epoch_stats_dict['valid']]

    train_MCC = [stat['MCC'] for stat in epoch_stats_dict['train']]
    valid_MCC = [stat['MCC'] for stat in epoch_stats_dict['valid']]

    train_diff = [stat['diff'] for stat in epoch_stats_dict['train']]
    valid_diff = [stat['diff'] for stat in epoch_stats_dict['valid']]

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes[0, 0].plot(epochs, train_loss, label='Train Loss')
    axes[0, 0].plot(epochs, valid_loss, label='Valid Loss')
    axes[0, 0].set_title('Loss per Epoch')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, train_accy, label='Train Accuracy')
    axes[0, 1].plot(epochs, valid_accy, label='Valid Accuracy')
    axes[0, 1].set_title('Accuracy per Epoch')
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, train_MCC, label='Train MCC')
    axes[1, 0].plot(epochs, valid_MCC, label='Valid MCC')
    axes[1, 0].set_title("Matthew's Correlation Coefficient (MCC) per Epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, train_diff, label='Train Diff')
    axes[1, 1].plot(epochs, valid_diff, label='Valid Diff')
    axes[1, 1].set_title('Diff per Epoch')
    axes[1, 1].legend()

    plt.tight_layout()
    
    if run is not None:

        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png')
        
        # Log the plot to Neptune
        run["plots/epoch_stats"].upload(tmpfile.name)
        plt.close(plt.gcf())
    else:

        plt.show()



















