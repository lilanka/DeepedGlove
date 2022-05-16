import json
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt

def softcopy(target, source, tau):
  """Copy parameters"""
  for target_param, param in zip(target, source):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def read_data(fname: str):
  """Read raw data file and take data"""
  data = []
  with open(fname) as f:
    data = [[float(y) for y in x.split()] for x in f.readlines()]
  return np.array(data), len(data[0])

def read_json(fname: str):
  """Read Json file and take data"""
  data = []
  with open(fname) as f:
    data = json.load(f)
  return data

def to_torch(array, copy=True, device='cpu'):
  """Convert a numpy array to Pytorch tensor"""
  if copy:
    return torch.tensor(array).to(device).float()
  return torch.as_tensor(array).to(device).float()

def debug(x, message):
  """For debuggin purposes"""
  print(f'Debug: {message}: {x}')

def equal(model1, model2):
  """Check if model1 and model2 are equal"""
  for p1, p2 in zip(model1.parameters(), model2.parameters()):
    if p1.data.ne(p2.data).sum() > 0:
      return False
  return True

def plot(x, y):
  """Plot graphs"""
  plt.plot(x, y)
  plt.show()
