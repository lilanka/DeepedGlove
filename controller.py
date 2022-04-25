import numpy as np

from utils import *
from buffer import Buffer
from model.simulator import Model 
from model.agent import Actor, Critic

class Controller():
  """
  Main controller of the system. Contains simmulator and agent.
  Args:
    configs: Configurations of the system. (Data in config.json)
    is_training: Training stage.
  """
  def __init__(self, configs, is_training=True): 

    self.device = configs["device"]
    self.buffer_size, self.batch_size = configs["buffer"]["size"], configs["buffer"]["batch_size"]
    self.action_dim = 50 # Assume total 50 motors that can be controllerd.

    self.actions = np.zeros((self.buffer_size, self.action_dim))  
    self.rewards = np.zeros((self.buffer_size, 1))
    n_costs = configs["costs"]["number_of_costs"]
    self.costs = np.zeros((self.buffer_size, n_costs))
    
    if is_training:
      self.data, self.obs_dim = read_data(configs["data"]["training"]) 
    else:
      self.data, self.obs_dim = read_data(configs["data"]["testing"])

    # Initialize buffer
    self.buffer = Buffer(self.buffer_size, self.data, self.actions, n_costs, configs["buffer"]["optimize_memory_usage"], self.device)
    self._initialize_buffer()

    # Initialize networks 
    self.actor = Actor(self.obs_dim, self.action_dim)
    self.reward_critic1 = Critic(self.obs_dim, self.action_dim)
    self.reward_critic2 = Critic(self.obs_dim, self.action_dim)
    self.cost_critic = Critic(self.obs_dim, self.action_dim)

    self.target_reward_critic1 = Critic(self.obs_dim, self.action_dim)
    self.target_reward_critic2 = Critic(self.obs_dim, self.action_dim)
    self.target_cost_critic = Critic(self.obs_dim, self.action_dim)

  def _initialize_buffer(self):
    """Initialize the buffer"""
    if self.buffer.size() == 0:
      for idx in range(self.buffer_size):
        self.buffer.add(self.data[idx], self.actions[idx], self.data[idx]+1, self.rewards[idx], self.costs[idx])  

  def train():
    """Train the system"""
    pass 

  def validate():
    """Validate the system"""
    pass

  def test():
    pass
