# %%
import numpy as np
import pandas as pd

# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
# 仅保留整点数据（分钟为00）
df = df[df['datetime'].dt.minute == 0].copy()
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.drop('datetime', axis=1))
test_scaled = scaler.transform(test.drop('datetime', axis=1))

# %%
# split X and y
def split_x_and_y(array, days_used_to_train=7):
    features = []
    labels = []
    for i in range(days_used_to_train, len(array)):
        features.append(array[i-days_used_to_train:i, :])
        labels.append(array[i, :])
    return np.array(features), np.array(labels)

train_X, train_y = split_x_and_y(train_scaled)
test_X, test_y = split_x_and_y(test_scaled)

# %%
# creat dataloaders
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_X_tensor = torch.FloatTensor(train_X).to(device)
train_y_tensor = torch.FloatTensor(train_y).to(device)
test_X_tensor = torch.FloatTensor(test_X).to(device)
test_y_tensor = torch.FloatTensor(test_y).to(device)

train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# build a LSTM model
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=None):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size if output_size else input_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.dense(lstm_out[:, -1, :])
        return output

input_size = train_X.shape[2]
output_size = train_y.shape[1]
model = LSTMModel(input_size=input_size, hidden_size=64, output_size=output_size).to(device)

# %%
# train the model
import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        train_loss += loss.item() * batch_x.size(0)
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}')

# %%
# evaluate the model on the test set
model.eval()
with torch.no_grad():
    pred_y = model(test_X_tensor).cpu().numpy()
    test_y_np = test_y_tensor.cpu().numpy()

# %%
# plotting the predictions against the ground truth
import matplotlib.pyplot as plt
num_dims = pred_y.shape[1] if pred_y.ndim > 1 else 1
plt.figure(figsize=(12, 2.5 * num_dims))
for i in range(num_dims):
    plt.subplot(num_dims, 1, i+1)
    plt.plot(pred_y[:, i] if num_dims > 1 else pred_y, label='Prediction', alpha=0.8)
    plt.plot(test_y_np[:, i] if num_dims > 1 else test_y_np, label='Ground Truth', alpha=0.8)
    plt.xlabel('Amount of samples')
    plt.ylabel(f'Dim {i}')
    plt.title(f'Prediction vs Ground Truth (Dim {i})')
    plt.legend(loc='upper right')
    plt.grid(True)
    # 只显示部分数据，避免曲线重叠太密
    if len(pred_y) > 500:
        plt.xlim(0, 500)
plt.tight_layout()
plt.show()

# %%
