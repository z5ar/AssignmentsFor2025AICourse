'''
Homework:
Methodology for solving image classification problems.
Train a simple convolutional neural network (CNN) to classify CIFAR images.
'''

# %%
# importing
import torch
import numpy as np
from matplotlib import pyplot as plt
# %%
# load CIFAR-10 data
from torchvision import datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
train_images, train_labels = train_dataset.data, np.array(train_dataset.targets)
test_images, test_labels = test_dataset.data, np.array(test_dataset.targets)
# %%
# Code here!
# inspect images
class_names = train_dataset.classes

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# %%
# prepocess dataset (include dataloader)
'''
1. one-hot encoded lavels:Transforming the label value into a 10 element binary vector   
(sometimes dont't needed, eg: when using "sparse_categorical_crossentropy" as the loss function in model.compile, we don't need to transfrom the labels, cause this loss function expects integer labels and it does the one-hot encoding it self(sparse_))  
2. Normalize the pixal data, scaling the pixel data from [0,255] to range [0,1]
'''
train_images = train_images/255.0
test_images = test_images/255.0

train_images=torch.tensor(train_images).permute(0,3,1,2).float()
test_images=torch.tensor(test_images).permute(0,3,1,2).float()
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

from torch.utils.data import DataLoader, TensorDataset
train_tensor_dataset = TensorDataset(train_images, train_labels)
test_tensor_dataset = TensorDataset(test_images, test_labels)
batch_size = 64
train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size,shuffle=False)

# %%
# create a CNN model
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*8*8,128)
        self.fc2=nn.Linear(128,10)
        self.relu=nn.ReLU()
        self.flatten=nn.Flatten()

    def forward(self, x):
        x=self.relu(self.conv1(x))
        x=self.pool(x)
        x=self.relu(self.conv2(x))
        x=self.pool(x)
        x=self.relu(self.conv3(x))
        x=self.flatten(x)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
# train the model
import torch.optim as optim
from torchmetrics import Accuracy

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
accuracy_metric = Accuracy(task="multiclass", num_classes=10).to(device)

def train_epoch(model, train_loader, optimizer, loss_fn, accuracy_metric,device):
    model.train()
    running_loss = 0.0
    running_acc= 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        running_acc += accuracy_metric(preds, labels) * images.size(0)
        
    return (running_loss / len(train_loader.dataset),
            running_acc / len(train_loader.dataset))

num_epochs = 12
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, accuracy_metric, device)
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"\tTrain Loss: {train_loss:.6f}\n\tTrain Accuracy: {train_acc:.4f}\n")


# %%
# evaluate the model
def test(model,dataloader, loss_fn, accuracy_metric, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_acc += accuracy_metric(preds, labels) * images.size(0)
    return (running_loss / len(dataloader.dataset),
            running_acc / len(dataloader.dataset))

test_loss, test_acc = test(model, test_loader, loss_fn, accuracy_metric, device)
print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.4f}")

probability_model=nn.Sequential(
    model,
    nn.Softmax(dim=1)
)

probability_model.eval()
with torch.no_grad():
    test_inputs = test_images.to(device)
    test_inputs = test_inputs.float()
    predictions = probability_model(test_inputs)
    predictions = predictions.cpu().numpy()

# %%
predictions[0]
# %%
predicted_class = np.argmax(predictions[0])
predicted_class
# %%
test_labels[0].item()
# %%
# visualization
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img.permute(1,2,0).numpy())
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
# %%
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# %%
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# %%
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
# %%
