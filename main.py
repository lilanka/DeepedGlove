#!/bin/python3

from utils import *
from controller import Controller

if __name__ == '__main__':
  configs = read_json("config.json")
  is_training = configs["is_training"]
  controller = Controller(configs, is_training)
  controller.validate()
  controller.test()
