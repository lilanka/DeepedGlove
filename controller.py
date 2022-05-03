import copy

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

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.buffer_size, self.batch_size = configs["buffer"]["size"], configs["buffer"]["batch_size"]
    self.action_dim = 50 # Assume total 50 motors that can be controlled.
   
    # These all are dummy data for the research
    self.obs_dim = 100
    n_costs = configs["costs"]["number_of_costs"]
    self.data = np.random.rand(self.buffer_size, 1, self.obs_dim)
    self.actions = np.zeros((self.buffer_size, 1, self.action_dim))  
    self.rewards = np.zeros((self.buffer_size, 1, 1))
    self.costs = np.zeros((self.buffer_size, 1, n_costs))
    
    """
    if is_training:
      self.data, self.obs_dim = read_data(configs["data"]["training"]) 
    else:
      self.data, self.obs_dim = read_data(configs["data"]["testing"])
    """
    # Initialize buffer
    self.buffer = Buffer(self.buffer_size, self.data[0], self.actions[0], n_costs, configs["buffer"]["optimize_memory_usage"], self.device)
    self._initialize_buffer()

    # Initialize networks 
    self.actor = Actor(self.obs_dim, self.action_dim, device=self.device)
    self.reward_critic1 = Critic(self.obs_dim, self.action_dim, device=self.device)
    self.reward_critic2 = Critic(self.obs_dim, self.action_dim, device=self.device)
    self.cost_critic = Critic(self.obs_dim, self.action_dim, device=self.device)

    self.target_reward_critic1 = Critic(self.obs_dim, self.action_dim, device=self.device)
    self.target_reward_critic2 = Critic(self.obs_dim, self.action_dim, device=self.device)
    self.target_cost_critic = Critic(self.obs_dim, self.action_dim, device=self.device)

    # Optimizers
    self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=configs["actor"]["optimizer"]["lr"])
    self.reward_critic1_optim = torch.optim.Adam(self.reward_critic1.parameters(), lr=configs["critic"]["optimizer"]["lr"])
    self.reward_critic2_optim = torch.optim.Adam(self.reward_critic2.parameters(), lr=configs["critic"]["optimizer"]["lr"])
    self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(), lr=configs["critic"]["optimizer"]["lr"])

    self.loss_fn = nn.MSELoss() # Not sure about which loss function 

    # Pre-train actor with real data
    self.train(True)   
    
    # Initialize target networks
    self.target_reward_critic1 = copy.deepcopy(self.reward_critic1)
    self.target_reward_critic2 = copy.deepcopy(self.reward_critic2)
    self.target_cost_critic = copy.deepcopy(self.cost_critic)

    # Simulator
    #self.simulator = Model(

    target_initialized = equal(self.target_reward_critic1, self.reward_critic1) == \
    equal(self.target_reward_critic2, self.reward_critic2) == equal(self.cost_critic, self.cost_critic)
    print('Target networks contain parameters initialized.' if target_initialized else 'Target networks contain parameters not initialized.')


  def _initialize_buffer(self):
    """Initialize the buffer"""
    if self.buffer.size() == 0:
      for idx in range(self.buffer_size):
        self.buffer.add(self.data[idx], self.actions[idx], self.data[idx], self.rewards[idx], self.costs[idx])  

  def train(self, pre_train=False):
    """Train the system"""
    if pre_train:
      pre_train_iter_n = self.configs["training"]["pre_train_iter"]
      for i in range(pre_train_iter_n):
        batch = self.buffer.sample(self.batch_size)
        for j in range(self.batch_size):
          self.actor_optim.zero_grad()
          self.reward_critic1_optim.zero_grad()
          self.reward_critic2_optim.zero_grad()
          self.cost_critic_optim.zero_grad()

          # Predictions 
          policy_prediction = self.actor(batch[0][j])
          reward_critic1_prediction = self.reward_critic1(batch[0][j])
          reward_critic2_prediction = self.reward_critic2(batch[0][j])
          cost_critic_prediction = self.cost_critic(batch[0][j])
          
          loss = self.loss_fn(policy_prediction, batch[1][j])
          loss.backward()
          self.actor_optim.step()
          
          # Add https://medium.com/bcggamma/lagrangian-relaxation-can-solve-your-optimization-problem-much-much-faster-daa9edc47cc9 
          # for optimization of Q1, Q2, Qc
          print(f'Iter number: {i+1}, Batch number {j+1}, Policy loss = {loss}')

    else:
      train_iter = self.configs["training"]["train_iter"]
      for i in range(train_iter):
        batch = self.buffer.sample(self.batch_size)
        self._restrictive_exploration(batch)
     
  def _restrictive_exploration(batch):
    # Threashholds
    lu, lp = self.configs["sim"]["lu"], self.configs["sim"]["lp"]
  

  def validate(self):
    """Validate the system"""
    pass

  def test(self):
    pass
