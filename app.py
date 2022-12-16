import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class RGBDataset(Dataset):
  def __init__(self, data_file):
    self.data = pd.read_csv(data_file)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    raw_value = self.data.iloc[idx, 0:3]
    raw_label = self.data.iloc[idx, 3]

    label = 0

    if raw_label == 'dark':
      label = 1

    return torch.tensor([raw_value[0], raw_value[1], raw_value[2]], dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(3, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        # nn.ReLU(),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits

training_data = RGBDataset("rgb_train.csv")
test_data = RGBDataset("rgb_test.csv")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

model = NeuralNetwork().to(device)
m = nn.Sigmoid()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# train_features, train_labels = next(iter(train_dataloader))
# print(train_labels)
# logits = model(train_features)
# pred = torch.round(m(logits))
# loss = loss_fn(pred.flatten(), train_labels)
# print(loss)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    logits = model(X)
    pred = m(logits)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # for name, param in model.named_parameters():
    #   print(name, param.grad)

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      logits = model(X)
      pred = m(logits)
      test_loss += loss_fn(pred, y).item()
      print(pred, y)
      correct += (pred == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def run():
  epochs = 5
  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
  print("Done!")

run()