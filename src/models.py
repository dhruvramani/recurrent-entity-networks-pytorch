import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RowCNN(nn.Module):
    def __init__(self, num_classes=8, split_size=100):
        super(RowCNN, self).__init__()
        self.window_sizes = [3, 4, 5]
        self.n_filters = 100
        self.num_classes = num_classes
        self.split_size = split_size
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.n_filters, [window_size, self.split_size], padding=(window_size - 1, 0))
            for window_size in self.window_sizes
        ])

        self.linear = nn.Linear(self.n_filters * len(self.window_sizes), self.num_classes)
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xs = []
        # x = torch.unsqueeze(x, 1) # Might have to do it - [B, CH, R, C]
        # TODO : Very unsure, lot of changes to do
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, R_, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, R_]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2) 
        x = x.view(x.size(0), -1)  
        logits = self.linear(x)
        logits = self.sigmoid(logits)
        return logits

class BiRNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=100, num_layers=2, num_classes=8):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):    
        # Aren't the weights of the hidden units, but just placeholders to store the hidden activation for the current run
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, (h0, c0) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.sigmoid(self.fc(out[:, -1, :]))
        return out

class RecurrentEntityNetwork(nn.Module):
    def __init__(self, num_blocks, num_units_per_block, keys, initializer=torch.randn, recurrent_initializer=torch.randn, activation=torch.nn.ReLU):
        self.num_blocks = num_blocks
        self.num_units_per_block = num_units_per_block
        self.keys = keys # entities
        self.initializer = initializer
        self.recurrent_initializer = recurrent_initializer
        self.activation= activation

        self.U = Variable(self.recurrent_initializer(self.num_units_per_block, self.num_units_per_block))
        self.V = Variable(self.recurrent_initializer(self.num_units_per_block, self.num_units_per_block))
        self.W = Variable(self.recurrent_initializer(self.num_units_per_block, self.num_units_per_block))
        self.U_bias = Variable(self.initializer(self.num_units_per_block))

    def state_size(self):
        return self.num_blocks * self.num_units_per_block

    def output_size(self):
        return self.num_blocks * self.num_units_per_block

    # NOTE : Change this
    def zero_state(self, batch_size, dtype):
        "Initialize the memory to the key values."
        zero_state = tf.concat([tf.expand_dims(key, axis=0) for key in self._keys], axis=1)
        zero_state_batch = tf.tile(zero_state, [batch_size, 1])
        return zero_state_batch

    def get_gate(self, state_j, key_j, inputs):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j)
        """
        a = torch.sum(inputs * state_j, axis = 1) 
        b = torch.sum(inputs * key_j, axis=1) 
        return torch.nn.sigmoid(a + b)

    def get_candidate(self, state_j, key_j, inputs, U, V, W, U_bias):
        """
        Represents the new memory candidate that will be weighted by the
        gate value and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        """
        key_V = torch.matmul(key_j, V) 
        state_U = torch.matmul(state_j, U) + U_bias
        inputs_W = torch.matmul(inputs, W)
        return self.activation(state_U + inputs_W + key_V)

    def forward(self, inputs, state, scope=None):
        # Split the hidden state into blocks (each U, V, W are shared across blocks).
        state = torch.split(state, self.num_blocks, dim=1)
        next_states = []
        for j, state_j in enumerate(state): # Hidden State (j)
            key_j = torch.unsqueeze(self.keys[j], dim=0) 
            gate_j = self.get_gate(state_j, key_j, inputs)
            candidate_j = self.get_candidate(state_j, key_j, inputs, self.U, self.V, self.W, self.U_bias)

            # Equation 4: h_j <- h_j + g_j * h_j^~
            # Perform an update of the hidden state (memory).
            state_j_next = state_j + torch.unsqueeze(gate_j, dim=-1) * candidate_j

            # Equation 5: h_j <- h_j / \norm{h_j}
            # Forget previous memories by normalization.
            state_j_next_norm = F.normalize(state_j_next, p=2, dim=-1)
            state_j_next_norm = torch.where(torch.gt(state_j_next_norm, 0.0), state_j_next_norm, torch.ones(state_j_next_norm.shape))
            state_j_next = state_j_next / state_j_next_norm

            next_states.append(state_j_next)
        state_next = torch.cat(next_states, axis=1)
        return state_next, state_next

