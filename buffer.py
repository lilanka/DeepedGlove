import random
import psutil
from typing import Union

import torch 
import numpy as np

from utils import *

class Buffer(object):
  """
  Class that represent a buffer
  Args:
    buffer_size: Maximum number of elements in the buffer
    obs_space: Observation space
    action_space: Action space
    device: Pytorch device to which the values will be converted
    n_costs: Number of cost functions
    otimize_mem_usage: https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

  Buffer has the type {'s','a','ns', 'r', 'c'}. 
  """

  def __init__(self, buffer_size, observation_space, action_space, n_costs, optimize_mem_usage, device: Union[torch.device, str] = 'cpu'):

    # Check that the replay buffer can fit into the memory
    if psutil is not None:
      mem_available = psutil.virtual_memory().available

    self.buffer_size = buffer_size
    self.observation_space = observation_space
    self.action_space = action_space
    self.obs_dim = observation_space.shape
    self.action_dim = action_space.shape
    self.n_costs = n_costs 
    self.device = device
    self.optimize_mem_usage = optimize_mem_usage

    self.end_position = 0
    self.is_buffer_full = False
   
    # TODO: Add dtype to reduce memory usage
    self.observations = np.zeros((self.buffer_size, self.obs_dim[0], self.obs_dim[1])) 
    self.actions = np.zeros((self.buffer_size, self.action_dim[0], self.action_dim[1]))
    self.rewards = np.zeros((self.buffer_size, 1))
    self.costs = np.zeros((self.buffer_size, n_costs))

    if optimize_mem_usage:
      # observations contains also the next observation
      self.next_observations = None
    else:
      self.next_observations = np.zeros(self.observations.shape)

    if psutil is not None:
      total_memory_usage = self.observations.nbytes + self.actions.nbytes +  self.rewards.nbytes + self.costs.nbytes
      
      if not self.optimize_mem_usage:
        total_memory_usage += self.next_observations.nbytes        

      if total_memory_usage > mem_available:
        warnings.warn(f'Not enough memory. Buffer size {total_memory_usage / 1e9}GB > Memory available {mem_available / 1e9}GB') 

  def size(self):
    """Return the current size of the buffer"""
    if self.is_buffer_full:
      return self.buffer_size
    return self.end_position

  def add(self, obs, action, next_obs, reward, cost):
    """Add elements to the buffer"""

    if self.is_buffer_full:
      print('The bufer is full')
      return None
    
    self.observations[self.end_position] = np.array(obs).copy()
    self.actions[self.end_position] = np.array(action).copy()
    self.rewards[self.end_position] = np.array(reward).copy()
    self.costs[self.end_position] = np.array(cost).copy()

    if self.optimize_mem_usage:
      self.observations[(self.end_position + 1) % self.buffer_size] = np.array(next_obs).copy()
    else:
      self.next_observations[self.end_position] = np.array(next_obs).copy()

    self.end_position += 1
    if self.end_position == self.buffer_size:
      self.is_buffer_full = True
      self.end_position = 0

  def reset(self):
    """Reset the buffer"""
    self.end_position = 0
    self.is_buffer_full = False

  def sample(self, n):
    """Sample n number of samples from the buffer"""
    if self.is_buffer_full:
      batch_idxs = (np.random.randint(1, self.buffer_size,  size=n) + self.end_position) % self.buffer_size
    else:
      batch_idxs = np.random.randint(0, self.end_position, size=n)

    if self.optimize_mem_usage:
      next_obs = self.observations[(batch_idxs + 1) % self.buffer_size, :]
    else:
      next_obs = self.observations[batch_idxs, :]

    data = (
      to_torch(self.observations[batch_idxs, :]),
      to_torch(self.actions[batch_idxs, :]),
      to_torch(next_obs),
      to_torch(self.rewards[batch_idxs].reshape(-1, 1)),
      to_torch(self.costs[batch_idxs])
    )
    return data 

""""
if __name__ == '__main__':
  buffer = Buffer(10, np.ones((1, 10)), np.ones((1, 10)), 10, True, device='cpu')

  obs = np.random.randint(1, 100, size=10)
  actions = np.random.randint(1, 100, size=10)
  next_obs = np.random.randint(1, 100, size=10)
  rewards = np.ones((10, 1))
  cost = np.ones((1, 10)) + 4.2
  
  buffer.add(obs, actions, next_obs, [1], cost)
  buffer.add(obs, actions, next_obs, [10], cost)
  buffer.add(obs, actions, next_obs, [2], cost)

  data = buffer.sample(1)
  debug(data, 'data')
"""
