import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, input_dim, hidden_size, tagset_size):
    super(Model, self).__init__()
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(input_dim, hidden_size)
    self.fc = nn.Linear(hidden_size, tagset_size) 

  def forward(self, x):
    """x is a tuple which has the shape (s, a)"""
    x = torch.cat((x[0], x[1]), -1)
    out, _ = self.lstm(x)
    out = F.relu(self.fc(out))
    return out

class Simulator(nn.Module):
  """Simulator simulates each section of the plant"""
  def __init__(self, configs, device='cpu'):
    super(Simulator, self).__init__()
    self.configs = configs
    self.hidden_size = configs["sim"]["args"]["hidden_units"]
    
    # Dipping tank
    self.dipping_temp_dim = configs["sim"]["dipping"]["n_temp"] 
    self.dipping_action_dim = configs["sim"]["dipping"]["n_motor"] 
    self.dipping = Model(self.dipping_temp_dim+self.dipping_action_dim, self.hidden_size, self.dipping_temp_dim)

    # Leaching tank
    self.leaching_temp_dim = configs["sim"]["leaching"]["n_temp"]
    self.leaching_action_dim = configs["sim"]["leaching"]["n_motor"] 
    self.leaching = Model(self.leaching_temp_dim+self.leaching_action_dim, self.hidden_size, self.leaching_temp_dim)

    # Oven
    self.oven_temp_dim = configs["sim"]["oven"]["n_temp"]
    self.oven_action_dim = configs["sim"]["oven"]["n_motor"] 
    self.oven = Model(self.oven_temp_dim+self.oven_action_dim, self.hidden_size, self.oven_temp_dim)

  def forward(self, x):
    """See the NOTE in README.md. x is a tupple as (s, a)"""
    dipping_prediction = self.dipping((x[0][:, 0, : self.dipping_temp_dim], x[1][:, 0, : self.dipping_action_dim])) 
    leaching_prediction = self.leaching((x[0][:, 0, self.dipping_temp_dim: self.dipping_temp_dim+self.leaching_temp_dim], \
                                  x[1][:, 0, self.dipping_action_dim: self.dipping_action_dim+self.leaching_action_dim])) 
    oven_prediction = self.oven((x[0][:, 0, self.dipping_temp_dim+self.leaching_temp_dim:], \
                          x[1][:, 0, self.dipping_action_dim+self.leaching_action_dim:]))

    out = torch.cat((dipping_prediction, leaching_prediction, oven_prediction), -1)
    return out
