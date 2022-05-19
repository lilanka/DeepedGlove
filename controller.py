import copy

import numpy as np
import torch.nn as nn 
import torch.nn.functional as F

from utils import *
from buffer import Buffer
from model.simulator import Simulator 
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
   
    # Dummy data for the research ------------------------------------
    self.obs_dim = 100 # Whole plant has combined 100 sensory observations
    self.action_dim = 50 # Assume total 50 motors that can be controlled.

    n_costs = configs["costs"]["number_of_costs"]
    self.data = np.random.rand(self.buffer_size, 1, self.obs_dim)
    self.actions = np.zeros((self.buffer_size, 1, self.action_dim))  
    self.rewards = np.zeros((self.buffer_size, 1, 1))
    self.costs = np.zeros((self.buffer_size, 1, n_costs))
    self.ro = 0.5
    # ------------------------------------------------------------------------------ 

    """
    if is_training:
      self.data, self.obs_dim = read_data(configs["data"]["training"]) 
    else:
      self.data, self.obs_dim = read_data(configs["data"]["testing"])
    """
    # Initialize buffer
    self.buffer = Buffer(self.buffer_size, self.data[0], self.actions[0], n_costs, configs["buffer"]["optimize_memory_usage"], self.device)
    self.simulator_buffer = Buffer(self.buffer_size, self.data[0], self.actions[0], n_costs, configs["buffer"]["optimize_memory_usage"], self.device)

    self._initialize_buffer()

    # Initialize networks 
    self.actor = Actor(self.obs_dim, self.action_dim, device=self.device)
    self.reward_critic1 = Critic(self.obs_dim, self.action_dim, device=self.device)
    self.reward_critic2 = Critic(self.obs_dim, self.action_dim, device=self.device)
    self.cost_critic = Critic(self.obs_dim, self.action_dim, device=self.device)

    self.target_reward_critic1 = Critic(self.obs_dim, self.action_dim, device=self.device)
    self.target_reward_critic2 = Critic(self.obs_dim, self.action_dim, device=self.device)
    self.target_cost_critic = Critic(self.obs_dim, self.action_dim, device=self.device)

    # Simulator
    self.simulator = Simulator(configs, self.device)

    # Optimizers
    self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=configs["actor"]["optimizer"]["lr"])
    self.reward_critic1_optim = torch.optim.Adam(self.reward_critic1.parameters(), lr=configs["critic"]["optimizer"]["lr"])
    self.reward_critic2_optim = torch.optim.Adam(self.reward_critic2.parameters(), lr=configs["critic"]["optimizer"]["lr"])
    self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(), lr=configs["critic"]["optimizer"]["lr"])

    self.gamma = configs["actor"]["discount_factor"] # discount factor
    self.lmult = configs["actor"]["lagrangian_multiplier"] # lagrangian multiplier 
    self.threshold = configs["costs"]["threshold"]

    # Pre-train actor with real data
    self.train(pre_train=False)   
    
    # Initialize target networks
    self.target_reward_critic1 = copy.deepcopy(self.reward_critic1)
    self.target_reward_critic2 = copy.deepcopy(self.reward_critic2)
    self.target_cost_critic = copy.deepcopy(self.cost_critic)

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

        self.actor_optim.zero_grad()

        # Predictions 
        policy_prediction = self.actor(batch[0])
        loss = F.mse_loss(policy_prediction, batch[1])
        loss.backward()
        self.actor_optim.step()

        print(f'Iter number: {i+1}, Policy loss = {loss}')
     
      torch.save(self.actor, 'modeldata/actor.pkl')
    else:
      loss = []
      train_iter = self.configs["training"]["train_iter"]
      for i in range(train_iter):
        batch = self.buffer.sample(self.batch_size)

        self.reward_critic2_optim.zero_grad()
        self.cost_critic_optim.zero_grad()

        # self._restrictive_exploration(batch) 

        # Just train the model without restrictive exploration methods
        target_policy = self.actor(batch[2])
        y = torch.min(self.target_reward_critic1(batch[2], target_policy), self.target_reward_critic2(batch[2], target_policy))
        z = self.target_cost_critic(batch[2], target_policy)
  
        Qr_target_v = batch[3] + self.gamma * y

        Qr1_v = self.reward_critic1(batch[0], batch[1])
        Qr1_loss = F.mse_loss(Qr1_v, Qr_target_v)

        Qr2_v = self.reward_critic2(batch[0], batch[1])
        Qr2_loss = F.mse_loss(Qr2_v, Qr_target_v)

        Qc_target_v = batch[4] + self.gamma * z
        Qc_v = self.cost_critic(batch[0], batch[1])
        Qc_loss = F.mse_loss(Qc_v, Qc_target_v)

        self.reward_critic1_optim.zero_grad()
        Qr1_loss.backward(retain_graph=True)
        self.reward_critic1_optim.step()

        self.reward_critic2_optim.zero_grad()
        Qr2_loss.backward(retain_graph=True)
        self.reward_critic2_optim.step()

        self.cost_critic_optim.zero_grad()
        Qc_loss.backward(retain_graph=True)
        self.cost_critic_optim.step()

        L = torch.tensor((torch.min(Qr1_v, Qr2_v) - self.lmult * (Qc_v - self.threshold)).mean().detach(), requires_grad=True)

        self.actor_optim.zero_grad()
        L.backward()
        self.actor_optim.step()

        print(f'Iter number: {i+1}, Qr1 loss = {Qr1_loss}, Qr2_loss = {Qr2_loss}, Qc_loss = {Qc_loss}, L={L}')
        loss.append(Qr1_loss.detach().numpy()) 

      plot(np.arange(train_iter), loss)

  def _restrictive_exploration(batch):
    """Restrictive exploration"""
    # Threashholds
    lu, lp = self.configs["sim"]["lu"], self.configs["sim"]["lp"]
    for i in range(self.batch_size):
      s1 = batch[0][i]

  def validate(self):
    """Validate the system"""
    loss = []
    val_iter = self.configs["training"]["val_iter"]
    for i in range(val_iter):
      batch = self.buffer.sample(self.batch_size)

      target_policy = self.actor(batch[2])
      y = torch.min(self.target_reward_critic1(batch[2], target_policy), self.target_reward_critic2(batch[2], target_policy))
      z = self.target_cost_critic(batch[2], target_policy)

      Qr_target_v = batch[3] + self.gamma * y
      Qc_target_v = batch[4] + self.gamma * z

      Qr1_v = self.reward_critic1(batch[0], batch[1])
      Qr1_loss = F.mse_loss(Qr1_v, Qr_target_v)

      Qr2_v = self.reward_critic2(batch[0], batch[1])
      Qr2_loss = F.mse_loss(Qr2_v, Qr_target_v)
      
      Qc_v = self.cost_critic(batch[0], batch[1])
      Qc_loss = F.mse_loss(Qc_v, Qc_target_v)
    
      print(f'Iter number: {i+1}, Qr1 loss = {Qr1_loss}, Qr2_loss = {Qr2_loss}, Qc_loss = {Qc_loss}')

  def test(self):
    pass
