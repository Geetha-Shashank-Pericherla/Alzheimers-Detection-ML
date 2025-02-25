from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns

# Testing Loop
def test_loop(model= model, test_data= test_data, criterion= criterion, device= device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():  # No gradient calculation for testing
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device) # Move data to device

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            test_loss += loss.item() # Sum up the loss

            # Compute accuracy
            outputs = torch.argmax(outputs, dim=1)  # Get the predicted class
            labels = torch.argmax(labels, dim=1)    # Get the true class
            test_acc += accuracy(outputs, labels)   # Sum up the accuracy

    # Compute average loss and accuracy
    test_loss /= len(test_data)
    test_acc /= len(test_data)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    

# Get true labels and predictions
true_labels = []
pred_labels = []

class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

model.eval()
with torch.no_grad():
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)  
        outputs = model(images)
        
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(outputs.cpu().numpy())
        
true_labels= np.argmax(true_labels, axis=1)
pred_labels= np.argmax(pred_labels, axis=1)

# Getting the classification report
report = classification_report(true_labels, pred_labels, target_names=class_names)
print(report)

# Compute and plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


from sklearn.metrics import roc_auc_score

true_labels = []
pred_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)  
        outputs = model(images)
        
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(outputs.cpu().numpy())
        
# Convert true labels to one-hot encoding
#true_labels_one_hot = torch.nn.functional.one_hot(torch.tensor(true_labels), num_classes=4).numpy()
true_labels = torch.argmax(torch.tensor(true_labels), dim=1).numpy()
pred_probs = torch.nn.functional.softmax(torch.tensor(pred_labels), dim=1).numpy()

from torchmetrics.classification import CalibrationError,MulticlassCalibrationError

ece = CalibrationError(n_bins=10, task="multiclass", num_classes=4 )
calibration_error = ece(torch.tensor(pred_probs), torch.tensor(true_labels))
print(f"Expected Calibration Error: {calibration_error:.4f}")

auc_score = roc_auc_score(true_labels, pred_probs, multi_class="ovr")
print(f"AUC Score: {auc_score:.4f}")