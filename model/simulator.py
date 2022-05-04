import math
import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, input_dim, hidden_size, tagset_size):
    super(Model, self).__init__()
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(input_dim, hidden_size)
    self.fc = nn.Linear(hidden_size, tagset_size) 

  def forward(self, x):
    """x is a tuple which has the shape (s, a)"""
    print(x)
    x = torch.cat((x[0], x[1]), -1)
    print(x)
    out, _ = self.lstm(x)
    out = self.fc(out)
    return out

class Simulator():
  """Simulator simulates each section of the plant"""
  def __init__(self, configs, device='cpu'):
    self.hidden_size = configs["sim"]["hidden_size"]
    self.configs = configs
    
    # Dipping tank 1
    self.dipping_tank1_temp_dim = configs["sim"]["dip1"]["n_temp"] 
    self.dipping_tank1_action_dim = configs["sim"]["dip1"]["n_motor"] 
    self.dipping_tank1 = Model(self.dipping_tank1_temp_dim + self.dipping_tank1_action_dim, self.hidden_size, self.dipping_tank1_temp_dim)

    # Dipping tank 2
    self.dipping_tank2_temp_dim = configs["sim"]["dip1"]["n_temp"]
    self.dipping_tank2_action_dim = configs["sim"]["dip1"]["n_motor"] 
    self.dipping_tank2 = Model(self.dipping_tank2_temp_dim + self.dipping_tank2_action_dim, self.hidden_size, self.dipping_tank2_temp_dim)

    # Leaching tank 1
    self.leaching_tank1_temp_dim = configs["sim"]["leach1"]["n_temp"]
    self.leaching_tank1_action_dim = configs["sim"]["leach1"]["n_motor"] 
    self.leaching_tank1 = Model(self.leaching_tank1_temp_dim + self.leaching_tank1_action_dim, self.hidden_size, self.leaching_tank1_temp_dim)

    # Leaching tank 2
    self.leaching_tank2_temp_dim = configs["sim"]["leach2"]["n_temp"]
    self.leaching_tank2_action_dim = configs["sim"]["leach2"]["n_motor"] 
    self.leaching_tank2 = Model(self.leaching_tank2_temp_dim + self.leaching_tank2_action_dim, self.hidden_size, self.leaching_tank2_temp_dim)

    # Oven 1
    self.oven1_temp_dim = configs["sim"]["oven1"]["n_temp"]
    self.oven1_action_dim = configs["sim"]["oven1"]["n_motor"] 
    self.oven1 = Model(self.oven1_temp_dim + self.oven1_action_dim, self.hidden_size, self.oven1_temp_dim)

    # Oven 2
    self.oven2_temp_dim = configs["sim"]["oven2"]["n_temp"]
    self.oven2_action_dim = configs["sim"]["oven2"]["n_motor"] 
    self.oven2 = Model(self.oven2_temp_dim + self.oven2_action_dim, self.hidden_size, self.oven2_temp_dim) 
  

  def forward(self, x):
    """See the NOTE in README.md. x is a tupple as (s, a)"""
    dipping_tank1_predicted_obs = self.dipping_tank1((x[0][:][:self.dipping_tank1_temp_dim], \
                                                      x[1][:][:self.dipping_tank1_action_dim]))
    dipping_tank2_predicted_obs = self.dipping_tank1((x[0][:][self.dippint_tank1_temp_dim: self.dipping_tank2_temp_dim], \
                                                      x[1][:][self.dippint_tank1_action_dim: self.dipping_tank1_action_dim]))



if __name__ == '__main__':
  batch_size, input_dim, output_dim, hidden_size = 5, 12, 32, 16
  lstm = Model(input_dim, hidden_size, output_dim) 

  s = torch.randn(batch_size, 1, input_dim-1)
  y = torch.randn(batch_size, 1, 1)
  out = lstm((s, y))
  print(out)
