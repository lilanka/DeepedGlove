import math
import torch
import torch.nn as nn

class MogrifierLSTMCell(nn.Module):
  """
  See https://arxiv.org/pdf/1909.01792.pdf 
  """
  def __init__(self, input_dim: int, hidden_size: int, mogrify_steps):
    super(MogrifierLSTMCell, self).__init__()
    self.mogrify_steps = mogrify_steps
    self.lstm = nn.LSTMCell(input_dim, hidden_size)
    self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_dim)])
    for i in range(1, mogrify_steps):
      if i % 2 == 0:
        self.mogrifier_list.extend([nn.Linear(hidden_size, input_dim)])
      else:
        self.mogrifier_list.extend([nn.Linear(input_dim, hidden_size)])

  def mogrify(self, x, h):
    for i in range(self.mogrify_steps):
      if (i + 1) % 2 == 0:
        h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
      else:
        x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x
    return x, h

  def forward(self, x, states):
    h_t, c_t = states
    x, h_t = self.mogrify(x, h_t)
    h_t, c_t = self.lstm(x, (h_t, c_t))
    return h_t, c_t

class Model(nn.Module):
  """
  Model is based on the MogrifierLSTM from DeepMind.
  Args:
    input_dim: Number of features in simulator input.
    hidden_size: Number of hidden layers.
    output_dim: Number of output features.
    mogrify_steps: See the paper.
    tie_weights: Embedding weights and output weights are tied.
  """
  def __init__(self, input_dim: int, hidden_size: int, output_dim: int, mogrifiy_steps: int, tie_weights: bool, dropout: float = 0.5):
    super(Model, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(output_dim, input_dim)
    self.lstm1 = MogrifierLSTMCell(input_dim, hidden_size, mogrify_steps)
    self.lstm2 = MogrifierLSTMCell(hidden_size, hidden_size, mogrify_steps)
    self.fc = nn.Linear(hidden_size, output_dim)
    self.drop = nn.Dropout(dropout)
    if tie_weights:
      self.fc.weight = self.embedding.weight
  
  def forward(self, x):
    embed, batch_size, length = self.embedding(x), x.shape[0], x[0].shape[0]
    h_1, c_1 = [torch.zeros(batch_size, self.hidden_size*4), torch.zeros(batch_size, self.hidden_size*4)]
    h_2, c_2 = [torch.zeros(batch_size, self.hidden_size*4), torch.zeros(batch_size, self.hidden_size*4)]
    hidden_states, outputs = [], []
    
    for step in range(length):
      y = self.drop(embed[:, step])
      h_1, c_1 = self.lstm1(x, (h_1, c_1))
      h_2, c_2 = self.lstm1(h_1, (h_2, c_2))
      out = self.fc(self.drop(h_2))
      hidden_states.append(h_2.unsqueeze(1))
      outputs.append(out.unsqueeze(1))

    hidden_states = torch.cat(hidden_states, dim=1)
    outputs = torch.cat(outputs, dim=1)
    return outputs, hidden_states

if __name__ == '__main__':
  input_dim = 512
  hidden_size = 512
  output_dim = 30
  batch_size = 4
  lr = 3e-3
  mogrify_steps = 5
  dropout = 0.5
  tie_weights = True
  betas = (0, 0.999)
  weight_decay = 2.5e-4
  clip_norm = 10

  model = Model(input_dim, hidden_size, output_dim, mogrify_steps, tie_weights, dropout)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e08, weight_decay=weight_decay)

  # seq of shape (batch_size, max_words)
  x = torch.LongTensor([[ 8, 29, 18,  1, 17,  3, 26,  6, 26,  5],
                          [ 8, 28, 15, 12, 13,  2, 26, 16, 20,  0],
                          [15,  4, 27, 14, 29, 28, 14,  1,  0,  0],
                          [20, 22, 29, 22, 23, 29,  0,  0,  0,  0]])
  outputs, hidden_states = model(x)
  print(outputs.shape)
  print(hidden_states.shape)

