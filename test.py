import torch
from torch import nn

m = nn.Sigmoid()

def normalize_rgb(rgb):
  # normalize the tensor values so that dark colors are max 0.5, and light colors are max 255
  v = torch.tensor([rgb[0], rgb[1], rgb[2]], dtype=torch.float32)
  v /= 255
  return v

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
  # normalize the inputs
  v = normalize_rgb(rgb)
  
  # run the data
  model = torch.load("models/rgb_nn.pth")
  logits = model(v)
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

