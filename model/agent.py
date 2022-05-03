import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicModel(nn.Module):
  def __init__(self, input_dim: int, output_dim: int, bias=True, device='cpu'):
    super(Agent, self).__init__()
    self.fc1 = nn.Linear(input_dim, 200, bias=bias).to(device)
    self.fc2 = nn.Linear(200, 200, bias=bias).to(device)
    self.fc3 = nn.Linear(200, 200, bias=bias).to(device)
    self.fc4 = nn.Linear(200, 200, bias=bias).to(device)
    self.fc5 = nn.Linear(200, output_dim).to(device)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    return x

class Actor(nn.Module):
  """Policy model"""
  def __init__(self, input_dim: int, output_dim: int, bias=True, device='cpu'):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(input_dim, 300, bias=bias).to(device)
    self.fc2 = nn.Linear(300, 300, bias=bias).to(device)
    self.fc3 = nn.Linear(300, output_dim, bias=bias).to(device)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = torch.tanh(self.fc3(x)) 
    return x

class Critic(nn.Module):
  """Reward and Cost critic networks"""
  def __init__(self, input_dim: int, output_dim: int, bias=True, device='cpu'):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(input_dim, 400, bias=bias).to(device)
    self.fc2 = nn.Linear(400, 400, bias=bias).to(device)
    self.fc3 = nn.Linear(400, output_dim, bias=bias).to(device)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return x
