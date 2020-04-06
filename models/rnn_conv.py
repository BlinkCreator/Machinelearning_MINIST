import torch.nn as nn
import torch
class ImageRNN(nn.Module):
	def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs, device):
		super(ImageRNN, self).__init__()
		self.device = device
		self.n_neurons = n_neurons   # Hidden
		self.batch_size = batch_size
		self.n_steps = n_steps    # 64
		self.n_inputs = n_inputs  # 28
		self.n_outputs = n_outputs  # 10
		# Basic RNN layer
		self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)
		# Followed by a fully connected layer
		self.FC = nn.Linear(self.n_neurons, self.n_outputs)

	def init_hidden(self, ):
		# (num_layers, batch_size, n_neurons)
		return (torch.zeros(1, self.batch_size, self.n_neurons)).to(self.device)

	def forward(self, X):
		# transforms X to dimensions: n_steps X batch_size X n_inputs
		# 28 * 64 * 28
		X = X.permute(1, 0, 2)
		self.batch_size = X.size(1)
		self.hidden = self.init_hidden()
		lstm_out, self.hidden = self.basic_rnn(X, self.hidden)
		out = self.FC(self.hidden)
		# Output from fully connected layer  directly
		return out.view(-1, self.n_outputs)  # batch_size X n_output