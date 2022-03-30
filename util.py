import json

def read_json(fname):
  data = {}
  try:
    with open(fname) as dfile:
      data = json.load(dfile)
  except FileNotFoundError:
    raise FileNotFoundError
  return data

def read_txt(fname):
  data = open(fname, 'r').read()
  chars = list(set(data))
  data_size, vocab_size = len(data), len(chars)
  return data, data_size, vocab_size
