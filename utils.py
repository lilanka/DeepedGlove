import json

def read_data(fname):
  data = []
  with open(fname) as f:
    data = [[float(y) for y in x.split()] for x in f.readlines()]
  return data, len(data[0])

def read_json(fname):
  data = []
  with open(fname) as f:
    data = json.load(f)
  return data
