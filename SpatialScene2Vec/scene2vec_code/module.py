import torch 
import torch.nn as nn
import torch.utils.data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class LayerNorm(nn.Module):
    def __init__(self,feature_dim,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter('gamma',self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta",self.beta)
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.gamma*(x-mean)/(std+self.eps)+self.beta

def get_activation_function(activation,context_str):
    if activation == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str+' '+activation))

class SingleFeedForwardNN(nn.Module):
    def __init__(self,input_dim,
                      output_dim,
                      dropout_rate=None,
                      activation="relu",
                      use_layernormalize=False,
                      skip_connection = False,
                      context_str=""):
        super(SingleFeedForwardNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None
        
        self.act = get_activation_function(activation=activation,context_str=context_str)
        if use_layernormalize:
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None
        
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = None

        self.linear = nn.Linear(self.input_dim,self.output_dim)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self,input_tensor):
        assert input_tensor.size()[-1] == self.input_dim

        input_tensor = input_tensor.to(device)
        output = self.linear(input_tensor)
        output = self.act(output)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.skip_connection:
            output = output + input_tensor
        if self.layernorm is not None:
            output = self.layernorm(output)
        return output

class MultiLayerFeedForwardNN(nn.Module):
    def __init__(self,input_dim,
                      output_dim,
                      num_hidden_layers=0,
                      dropout_rate=None,
                      hidden_dim=128,
                      activation='relu',
                      use_layernormalize=False,
                      skip_connection=False,
                      context_str=None):
        super(MultiLayerFeedForwardNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str
        self.layers = nn.ModuleList()
        if self.num_hidden_layers<=0:
            self.layers.append(SingleFeedForwardNN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                dropout_rate=self.dropout_rate,
                activation = self.activation,
                use_layernormalize = False,
                skip_connection = False,
                context_str = context_str
            ))
        else:
            self.layers.append(SingleFeedForwardNN(
                input_dim=self.input_dim,
                output_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                activation = self.activation,
                use_layernormalize = self.use_layernormalize,
                skip_connection = self.skip_connection,
                context_str = context_str
            ))
            for i in range(self.num_hidden_layers-1):
                self.layers.append(SingleFeedForwardNN(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    activation = self.activation,
                    use_layernormalize = self.use_layernormalize,
                    skip_connection = self.skip_connection,
                    context_str = context_str
                ))
            self.layers.append(SingleFeedForwardNN(
                input_dim=self.hidden_dim,
                output_dim=self.output_dim,
                dropout_rate=self.dropout_rate,
                activation = self.activation,
                use_layernormalize = False,
                skip_connection = False,
                context_str = context_str
            ))
    
    def forward(self,input_tensor):
        assert input_tensor.size()[-1] == self.input_dim
        input_tensor = input_tensor.to(device)
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)
        return output

