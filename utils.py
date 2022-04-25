import json

import torch
import numpy as np

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

def to_torch(array: np.ndarray, copy: bool = True) -> torch.Tensor:
  """Convert a numpy array to Pytorch tensor"""
  if copy:
    return torch.tensor(array).to(self.device)
  return torch.as_tensor(array).to(self.device)

def debug(x, message):
  """For debuggin purposes"""
  print(f'Debug: {message}: {x}')
