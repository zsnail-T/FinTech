import torch
from torch import nn, optim
import torch.nn.functional as F

class LongShortTermMemory(nn.Module):
    def __init__(self, timestep, n_features, hidden_size, n_layers):
        super(LongShortTermMemory, self).__init__()
        
        self.hidden_size = hidden_size
        self.timestep = timestep
        self.n_layers = n_layers
        
        # built LSTM model
        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            bias=True,
                            batch_first=False,
                            dropout=0.5,
                            bidirectional=False
                           )
        self.num_directions = 2 if self.lstm.bidirectional else 1
        self.fc = nn.Linear(hidden_size*self.num_directions, 1)
        self.dropout = nn.Dropout(0.2)
        # attention net parameter
        self.atten_w = nn.Parameter(torch.Tensor(hidden_size*self.num_directions, hidden_size*self.num_directions))
        self.atten_u = nn.Parameter(torch.Tensor(hidden_size*self.num_directions, 1))
        
        # init parameters
        nn.init.uniform_(self.atten_w, -0.1, 0.1)
        nn.init.uniform_(self.atten_u, -0.1, 0.1)
        
    def attention_net(self, lstm_output, hidden_state):
#         u: [batch_size, num_features, 2 * num_hiddens]
        u = torch.relu(torch.matmul(lstm_output, self.atten_w))
        
        attention = torch.matmul(u, self.atten_u)
       # attention: [batch_size, attention, 1]
        attention_weight = F.softmax(attention, dim=1)
       # att_score: [batch_size, n_features, 1]
        context = lstm_output * attention_weight
        # context: [bath_size, hidden_size*self.num_directions]
        context = torch.sum(context, dim=1)

        return context, attention_weight
        
    
    def forward(self, x):
        x = x.permute(1,0,2)                  #[batch_size, timestep, n_features] -> [timestep, batch_size, n_features]
        
        # output: [timestep, batch_size, hidden_size * num_directions], hidden_state: [num_directions * num_layers, batch, hidden_size]
        output, (hidden_state, cell_state) = self.lstm(x)
#         print("LSTM_out:", output.shape, hidden_state.shape, cell_state.shape)
        output = output.permute(1, 0, 2)                  #[batch_size, timestep, hidden_size * num_directions]
        # attn_output: [batch_size, hidden_size], attention_wright: [batch_size, timestep, 1]
        attn_output, attention_weight = self.attention_net(output, hidden_state)
        logit = self.fc(attn_output)
#         print("logit:", logit.shape)
        logit = self.dropout(logit)
    
        return logit, attention_weight