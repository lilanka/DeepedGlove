#!/bin/python3

from utils import *
from controller import Controller

if __name__ == '__main__':
  configs = read_json("config.json")
  is_training = configs["is_training"] == "true"
  controller = Controller(configs, is_training)

  """
  controller.train()
  controller.validate()
  controller.test()
  """
