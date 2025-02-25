import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import Accuracy
from config import Config  # Import hyperparameters from config.py
from dataset import get_dataloaders  # Load train & val dataloaders
from model import MyModel  # Import your model
import os

torch.manual_seed(42)


from torch import optim
torch.manual_seed(42)

# CrossEntropyLoss (for multi-class classification)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Adam optimizer with weight decay (L2 regularization)
AdamW = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler (optional, helps with convergence)
scheduler = optim.lr_scheduler.CosineAnnealingLR(AdamW, T_max=40, eta_min=1e-6)


accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

def training_loop(model= model, num_epochs= 1, train_data= train_data, val_data= val_data, criterion= criterion, optimizer= AdamW, scheduler= scheduler, device= device, use_early_stopping=False, patience=5):
    train_losses, val_losses = [], []            # Stores train data metrics
    train_accuracies, val_accuracies = [], []    # Stores validation data metrics

    for epoch in range(num_epochs):
        model.train()                          # Set the model to training mode
        train_loss, train_acc = 0.0, 0.0       # Initialize metrics for the epoch

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)   # Move data to the device

            optimizer.zero_grad()                # Reset gradients
            outputs = model(images)              # forward pass
            loss = criterion(outputs, labels)    # Compute loss
            loss.backward()                      # backpropagation
            optimizer.step()                     # update weights

            train_loss += loss.item()            # Accumulate loss

            # Compute accuracy
            outputs = torch.argmax(outputs, dim=1)        # Convert to class labels
            labels = torch.argmax(labels, dim=1)          # Convert to class labels
            train_acc += accuracy(outputs, labels)        # Compute accuracy and accumulate it

        # Compute training average loss and accuracy for the epoch
        train_loss /= len(train_data)
        train_acc /= len(train_data)
        
        # Store training loss and accuracy for plotting
        train_losses.append(train_loss)                
        train_accuracies.append(train_acc.cpu().item())  

        # Validation Step
        model.eval()                       # Set the model to evaluation mode
        val_loss, val_acc = 0.0, 0.0       # Initialize metrics for the epoch
        with torch.no_grad():                 # Disable gradient computation
            for images, labels in val_data:
                
                # Compute val loss                     
                images, labels = images.to(device), labels.to(device)   
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Compute val accuracy
                val_loss += loss.item()
                outputs = torch.argmax(outputs, dim=1)
                labels = torch.argmax(labels, dim=1)
                val_acc += accuracy(outputs, labels)

        # Compute validation average loss and accuracy for the epoch
        val_loss /= len(val_data)
        val_acc /= len(val_data)
        
        # Store validation loss and accuracy for plotting
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.cpu().item())

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early Stopping Check
        if use_early_stopping:
            if val_loss < best_val_loss:   
                best_val_loss = val_loss      # Update best validation loss
                early_stopping_counter = 0    # Reset counter
            else:                                        
                early_stopping_counter += 1   # Increment counter
                print(f"Early stopping patience: {early_stopping_counter}/{patience}")
                
                if early_stopping_counter >= patience: # Check if patience is reached
                    print("Early stopping triggered. Training stopped.")
                    break                              # Stop training

        # Step LR scheduler (if used)
        scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies

# Main script execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, val_loader = get_dataloaders(Config.batch_size)

    # Initialize model
    model = MyModel().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train the model
    training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device)