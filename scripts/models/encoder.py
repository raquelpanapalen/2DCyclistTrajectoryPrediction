import torch.nn as nn

# Define the encoder module
class Encoder(nn.Module):
    '''Encodes time-series sequence'''

    def __init__(self, input_size, hidden_size, device, num_layers=1, drop_prob=0.5):
        '''
        param input_size:     the number of features in the input X
        param hidden_size:    the number of features in the hidden state h
        param num_layers:     number of recurrent layers (i.e., 2 means there are 2 stacked LSTMs)
        '''
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        # define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input):
        '''
        input: (batch_size, seq_length, input_size)

        return:
          - lstm_out: (batch_size, seq_length, hidden_size) --> gives all the hidden states in the sequence
          - hidden: (num_layers, batch_size, hidden_size) --> gives the hidden state and the cell state for the last element in the sequence
        '''
        lstm_out, self.hidden = self.lstm(input)
        out = self.dropout(lstm_out)
        return out, self.hidden

    def init_hidden(self, batch_size):
        '''Initializes hidden state'''
        # Create two new tensors with sizes num_layers x batch_size x hidden_size,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        self.hidden = (
            weight.new(self.num_layers, batch_size, self.hidden_size)
            .zero_()
            .to(self.device),
            weight.new(self.num_layers, batch_size, self.hidden_size)
            .zero_()
            .to(self.device),
        )

        return self.hidden