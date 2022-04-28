import numpy as np
import torch.nn as nn 

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
    self.configs = configs
    self.device = configs["device"]
    self.buffer_size, self.batch_size = configs["buffer"]["size"], configs["buffer"]["batch_size"]
    self.action_dim = 50 # Assume total 50 motors that can be controlled.
   
    # These all are dummy data for the research
    self.obs_dim = 100
    self.data = np.random.rand(1000, self.obs_dim)
    self.actions = np.zeros((self.buffer_size, self.action_dim))  
    self.rewards = np.zeros((self.buffer_size, 1))
    n_costs = configs["costs"]["number_of_costs"]
    self.costs = np.zeros((self.buffer_size, n_costs))
    
    """
    if is_training:
      self.data, self.obs_dim = read_data(configs["data"]["training"]) 
    else:
      self.data, self.obs_dim = read_data(configs["data"]["testing"])
    """
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

    # Optimizers
    self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=configs["actor"]["optimizer"]["lr"])
    self.reward_critic1_optim = torch.optim.Adam(self.reward_critic1.parameters(), lr=configs["critic"]["optimizer"]["lr"])
    self.reward_critic2_optim = torch.optim.Adam(self.reward_critic2.parameters(), lr=configs["critic"]["optimizer"]["lr"])
    self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(), lr=configs["critic"]["optimizer"]["lr"])

    self.loss_fn = nn.MSELoss() # Not sure about which loss function 

  def _initialize_buffer(self):
    """Initialize the buffer"""
    if self.buffer.size() == 0:
      for idx in range(self.buffer_size):
        self.buffer.add(self.data[idx], self.actions[idx], self.data[idx], self.rewards[idx], self.costs[idx])  

  def train(self, pre_train=False):
    """Train the system"""
    if pre_train:
      pre_train_iter = self.configs["training"]["pre_train_iter"]
      for i in range(pre_train_iter):
        batch = self.buffer.sample(self.batch_size)
        for j in range(self.batch_size):
          self.actor_optim.zero_grad()
          policy_prediction = self.actor(batch[0])
          loss = self.loss_fn(policy_prediction, batch[1])
          loss.backward()
          self.actor_optim.step()

          print(f'Iter number: {i+1}, Batch number {j+1}, loss = {loss}')

  def validate(self):
    """Validate the system"""
    pass

  def test(self):
    pass
