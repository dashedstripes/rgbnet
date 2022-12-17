import torch
from torch import nn

m = nn.Sigmoid()

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits

def run(rgb):
  model = torch.load("models/rgb_nn.pth")
  value = torch.tensor(rgb, dtype=torch.float32)
  logits = model(value)

  pred = torch.round(m(logits))

  if(pred == 0):
    return "Light"
  else:
    return "Dark"

print(run([242, 143, 136])) # light
print(run([48, 12, 9])) # dark
print(run([149, 44, 201])) # light
print(run([46, 30, 35])) # dark

print(run([234, 176, 255])) # light
print(run([84, 84, 84])) # dark

