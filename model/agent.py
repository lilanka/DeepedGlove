import torch
import torch.nn as nn

class DynamicModel(nn.Module):
  def __init__(self, input_dim: int, output_dim: int, bias=True):
    super(Agent, self).__init__()
    self.fc1 = nn.Linear(input_dim, 200, bias=bias)
    self.fc2 = nn.Linear(200, 200, bias=bias)
    self.fc3 = nn.Linear(200, 200, bias=bias)
    self.fc4 = nn.Linear(200, 200, bias=bias)
    self.fc5 = nn.Linear(200, output_dim)

  def forward(self, x):
    x = nn.ReLU(self.fc1(x))
    x = nn.ReLU(self.fc2(x))
    x = nn.ReLU(self.fc3(x))
    x = nn.ReLU(self.fc4(x))
    x = nn.ReLU(self.fc5(x))
    return x

class Agent(nn.Module):
  """Policy model"""
  def __init__(self, input_dim: int, output_dim: int, bias=True):
    super(Agent, self).__init__()
    self.fc1 = nn.Linear(input_dim, 300, bias=bias)
    self.fc2 = nn.Linear(300, 300, bias=bias)
    self.fc3 = nn.Linear(300, output_dim, bias=bias)

  def forward(self, x):
    x = nn.ReLU(self.fc1(x))
    x = nn.ReLU(self.fc2(x))
    x = torch.tanh(self.fc3(x)) 
    return x

class Critic(nn.Module):
  """Reward and Cost critic networks"""
  def __init__(self, input_dim: int, output_dim: int, bias=True):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(input_dim, 400, bias=bias)
    self.fc2 = nn.Linear(400, 400, bias=bias)
    self.fc3 = nn.Linear(400, output_dim, bias=bias)
  
  def forward(self, x):
    x = self.ReLU(self.fc1(x))
    x = self.ReLU(self.fc2(x))
    x = self.ReLU(self.fc3(x))
    return x
