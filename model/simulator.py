import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
  def __init__(self, input_dimension, output_dimension, neurons, num_layers=1, label_width=None): 
    super(LSTM, self).__init__()

    self.label_width = label_width
    self.lstm = nn.LSTM(input_dimension, neurons, batch_first=True, num_layers=num_layers)
    self.fc = nn.Linear(neurons, output_dimension)

  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    predictions = self.fc(lstm_out)
    if self.label_width is not None:
      return predictions[:, -self.label_width :]
    else:
      return predictions
