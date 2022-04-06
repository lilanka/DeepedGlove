import json

def read_data(fname):
  data = []
  with open(fname) as f:
    data = [x.split() for x in f.readlines()]
  return data

def read_json(fname):
  data = []
  with open(fname) as f:
    data = json.load(f)
  return data
