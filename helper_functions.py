import torch
import numpy as np

from matplotlib import pyplot as plt
from sklearn import metrics
from pathlib import Path


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    # Plots decision boundaries of model predicting on X in comparison to y
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c='b', label='Train Data')

  # Plot testing data in green
  plt.scatter(test_data, test_labels, c='g', label='Test Data')

  # Are there predictions?
  if predictions is not None:
    # Plot the predictions if they exist
    plt.scatter(test_data, predictions, c='r', label='Predictions')
  
  plt.legend(prop={'size': 14})


def visualize_loss(epoch_count_list, train_loss_list, test_loss_list):
  plt.figure(figsize=(10, 7))

  # Plot Training & Testing Loss
  plt.plot(epoch_count_list, train_loss_list, label='Train Loss')
  plt.plot(epoch_count_list, test_loss_list, label='Test Loss')
  plt.title('Train & Test Loss Curves')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(prop={'size': 14})
    
    
def visualize_accuracy(epoch_count_list, train_acc_list, test_acc_list):
  plt.figure(figsize=(10, 7))

  # Plot Training & Testing Accuracy
  plt.plot(epoch_count_list, train_acc_list, label='Train Accuracy')
  plt.plot(epoch_count_list, test_acc_list, label='Test Accuracy')
  plt.title('Train & Test Accuracy Curves')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(prop={'size': 14})


def save_model(model, model_name):
  # 1. Create models directory
  MODEL_PATH = Path('models')
  MODEL_PATH.mkdir(parents=True, exist_ok=True)

  # 2. Create model save path
  MODEL_NAME = model_name+'.pth'
  MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

  # 3. Save the model state dict
  torch.save(model.state_dict(), MODEL_SAVE_PATH)
  
    
# Evaluation Function
def accuracy_func(y_pred, y_true):
  correct = torch.eq(y_true, y_pred).sum().item()
  return ((correct/len(y_true))*100)

def precision_recall_f1Score(y_pred, y_true):
  precision = metrics.precision_score(y_true, y_pred) # tp / (tp + fp)
  recall = metrics.recall_score(y_true, y_pred) # tp / (tp + fn)
  f1_score =  metrics.f1_score(y_true, y_pred) # 2 * (precision * recall) / (precision + recall)
  return precision, recall, f1_score

def classification_report(y_pred, y_true):
  report = metrics.classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
  return report
