import torch
from torch import nn, optim
import torch.nn.functional as F

class LongShortTermMemory(nn.Module):
    def __init__(self, timestep, n_features, hidden_size, n_layers, n_child_features):
        super(LongShortTermMemory, self).__init__()
        
        self.hidden_size = hidden_size
        self.timestep = timestep
        self.n_layers = n_layers
        self.n_child_features = n_child_features
        
        # aggregation child features(Open, Close, High, Close, AdjClose)
        self.aggregate = nn.ModuleList()
        for i in range(10):
            self.aggregate.append(
                nn.Sequential(
                    nn.Linear(n_child_features, 500),
                    nn.BatchNorm1d(timestep),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(500, 1),
                    nn.BatchNorm1d(timestep),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2)
                )
            )
        # number of aggregated features
        n_features = n_features - (n_child_features-1) * 10
        
        # built LSTM model
        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            bias=True,
                            batch_first=True,
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
        
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, bsz, self.hidden_size),
                weight.new_zeros(self.n_layers, bsz, self.hidden_size))
        
    def aggregation_net(self, inputs):
        # inputs: [batch_size, timestep, n_features] (500, 100, 51)
        # outputs: [batch_size, timestep, n_features/5] (500, 100, 11)
        n_child_features = self.n_child_features
        n_features = inputs.size(-1)
        out = []
#         print(self.n_features, n_child_features)
        for i in range(10):
            out.append(self.aggregate[i](inputs[:, :, i*n_child_features:(i+1)*n_child_features]))
        # increasing sentiment to aggregated features
        out.append(inputs[:, :, 10*n_child_features:])
#         print(len(out))

        return torch.cat(out, dim=-1)
        
    def attention_net(self, lstm_output, hidden_state):
        # u: [batch_size, num_features, 2 * num_hiddens]
        u = torch.matmul(lstm_output, self.atten_w)
        
        attention = torch.matmul(u, self.atten_u)
       # attention: [batch_size, attention, 1]
        attention_weight = F.softmax(attention, dim=1)
       # att_score: [batch_size, n_features, 1]
        context = lstm_output * attention_weight
        # context: [bath_size, hidden_size*self.num_directions]
        context = torch.sum(context, dim=1)

        return context, attention_weight
    
    def forward(self, x):
        self.hidden = self.init_hidden(x.size(0))
        
        # aggregation child features
        x = self.aggregation_net(x)
        
#         x = x.permute(1,0,2)                  #[batch_size, timestep, n_features] -> [timestep, batch_size, n_features]
        
        # output: [timestep, batch_size, hidden_size * num_directions], hidden_state: [num_directions * num_layers, batch, hidden_size]
        output, self.hidden = self.lstm(x, self.hidden)
#         print("LSTM_out:", output.shape, hidden_state.shape, cell_state.shape)
#         output = output.permute(1, 0, 2)                  #[batch_size, timestep, hidden_size * num_directions]
        # attn_output: [batch_size, hidden_size], attention_wright: [batch_size, timestep, 1]
        attn_output, attention_weight = self.attention_net(output, self.hidden[0])
        logit = self.fc(attn_output)
#         print("logit:", logit.shape)
#         logit = nn.Tanh()(logit)
#         logit = self.dropout(logit)
    
        return logit, attention_weight