#!/bin/python3

from utils import *
from model.simulator import LSTM 

class Controller():
  def __init__(self, in_dimension, out_dimension = 10): 
    # Simulation model
    # Polymer tank model, Leaching tank model, Heating/Oven model 
    self.polymer_tank_model = LSTM(in_dimension, out_dimension, 5)
    self.leaching_tank_model = None
    self.oven_model = None 

if __name__ == '__main__':

  # data directories
  training_data_dir = 'data/training_data.txt'
  validation_data_dir = 'data/validation_data.txt'
  testing_data_dir = 'data/testing_data.txt'
 
  training_data, in_dimension = read_data(training_data_dir)

  x = Controller(20, 20)
