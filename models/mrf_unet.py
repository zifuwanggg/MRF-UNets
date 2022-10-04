import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ops import Down, Up, Cat, Head, Filter, ChoiceBlock


class MRFSuperNet(nn.Module):
    def __init__(self, image_channels, num_classes, channel_step):
        super(MRFSuperNet, self).__init__()

        self.supernet = SuperNet(image_channels, num_classes, channel_step)
        
        self.variables = list(range(self.supernet.layers + 1))
        self.num_states = dict()
        self.neighbors = dict()

        self.unary_potentials = nn.ParameterDict()
        self.pairwise_potentials = nn.ParameterDict()
        self.init_unary_potentials()
        self.init_pairwise_potentials()
        
        self.sort_neighbors()
                
        self._potentials = []
        for n, p in self.named_parameters():
            if 'potentials' in n:
                self._potentials.append((n, p))
    
    
    def forward(self, x, choices_one_hot):            
        return self.supernet(x, choices_one_hot)
               
                
    def weights(self):
        return self.supernet.parameters()


    def named_weights(self):
        return self.supernet.named_parameters()


    def potentials(self):
        for _, p in self._potentials:
            yield p


    def named_potentials(self):
        for n, p in self._potentials:
            yield n, p


    def edge_to_string(self, edge, normal=True):
        if normal: 
            return str(edge[0]) + ',' + str(edge[1])
        else: 
            return str(edge[1]) + ',' + str(edge[0])
               
                
    def set_unary_potentials(self, variable, potential):
        self.unary_potentials[str(variable)] = potential
        self.neighbors[variable] = set()
        self.num_states[variable] = potential.size()[0]


    def set_pairwise_potentials(self, edge, potential):
        assert potential.size() == (len(self.unary_potentials[str(edge[0])]), len(self.unary_potentials[str(edge[1])])), \
            "potential size %d, %d incompatible with unary sizes %d, %d" % \
            (potential.size()[0], potential.size()[1], len(self.unary_potentials[str(edge[0])]),
             len(self.unary_potentials[str(edge[1])]))

        if edge[0] < edge[1]:
            string = self.edge_to_string(edge, normal=True)
            self.pairwise_potentials[string] = potential
        else:
            string = self.edge_to_string(edge, normal=False)
            self.pairwise_potentials[string] = potential.t()

        self.neighbors[edge[0]].add(edge[1])
        self.neighbors[edge[1]].add(edge[0])


    def init_unary_potentials(self):
        for l in range(self.supernet.layers + 1):
            if l in [0, self.supernet.layers]:
                unary = nn.Parameter(torch.ones(1))
            elif (l - 1) in (self.supernet.up_idx + self.supernet.down_idx):
                unary = nn.Parameter(torch.ones(self.supernet.channel_step))
            else:
                unary = nn.Parameter(torch.ones(self.supernet.cardinality))
            self.set_unary_potentials(l, unary)


    def init_pairwise_potentials(self):
        for l in range(self.supernet.layers):   
            if l == 0:
                pairwise = nn.Parameter(torch.ones((1, self.supernet.cardinality)))
            elif l == self.supernet.layers - 1:
                pairwise = nn.Parameter(torch.ones((self.supernet.cardinality, 1)))
            elif l in self.supernet.down_idx:
                pairwise = nn.Parameter(torch.ones((self.supernet.cardinality, self.supernet.channel_step)))
            elif (l - 1) in self.supernet.down_idx:
                pairwise = nn.Parameter(torch.ones((self.supernet.channel_step, self.supernet.cardinality)))
            elif l in self.supernet.up_idx:
                pairwise = nn.Parameter(torch.ones((self.supernet.cardinality, self.supernet.channel_step)))
            elif (l - 1) in self.supernet.up_idx:
                pairwise = nn.Parameter(torch.ones((self.supernet.channel_step, self.supernet.cardinality)))
            else:
                pairwise = nn.Parameter(torch.ones((self.supernet.cardinality, self.supernet.cardinality)))
            
            edge = (l, l + 1)
            self.set_pairwise_potentials(edge, pairwise)
           
            if (l - 2) in self.supernet.up_idx:
                idx = self.supernet.step - self.supernet.up_idx.index(l-2) - 1
                idx = self.supernet.down_idx[idx]
                edge = (idx, l)
                pairwise = nn.Parameter(torch.ones((self.supernet.cardinality, self.supernet.cardinality)))
                self.set_pairwise_potentials(edge, pairwise)
    
    
    def sort_neighbors(self):
        for variable in self.variables:
            neighbors = list(self.neighbors[variable])
            neighbors.sort()
            self.neighbors[variable] = neighbors
            
            
    def get_pairwise_potentials(self, variable, neighbor):
        edge = (variable, neighbor)
        if variable < neighbor:
            string = self.edge_to_string(edge, normal=True)
            return self.pairwise_potentials[string]
        else:
            string = self.edge_to_string(edge, normal=False)
            return self.pairwise_potentials[string].T


    def create_factor_prod(self):
        factor_prod = {}

        for variable in self.variables:
            cardinality = len(self.neighbors[variable])
            unary = torch.exp(self.unary_potentials[str(variable)])

            if cardinality == 1: 
                factor_prod[variable] = unary
                continue

            for i, neighbor in enumerate(self.neighbors[variable]):  
                pairwise = torch.exp(self.get_pairwise_potentials(variable, neighbor))
                if i == 0: 
                    x0 = torch.einsum('i,ij->ij', unary, pairwise)
                elif i == 1: 
                    x1 = torch.einsum('ij,ik->ijk', x0, pairwise)
                    if cardinality == 2:
                        factor_prod[variable] = x1.to('cpu')
                else:
                    assert cardinality == 3
                    x2 = torch.einsum('ijk,il->ijkl', x1, pairwise)
                    factor_prod[variable] = x2.to('cpu')
        
        return factor_prod


    def burnin(self, factor_prod, length, choices=None):
        if choices is None:
            choices = torch.zeros(len(self.variables), dtype=torch.long, requires_grad=False)
            for variable in self.variables:
                prob = torch.ones(self.num_states[variable], requires_grad=False)
                index = torch.multinomial(prob, 1)
                choices[variable] = index

        with torch.no_grad():
            for _ in range(length):
                for variable in self.variables:
                    neighbors = list(self.neighbors[variable])
                    cardinality = len(neighbors)
                    if cardinality == 1:
                        index = 0
                    elif cardinality == 2:
                        prob = factor_prod[variable][:, choices[neighbors[0]], choices[neighbors[1]]]
                        index = torch.multinomial(prob, 1)
                    elif cardinality == 3:
                        prob = factor_prod[variable][:, choices[neighbors[0]], choices[neighbors[1]], choices[neighbors[2]]]
                        index = torch.multinomial(prob, 1)
                    choices[variable] = index
        
        return choices


    def sample(self, factor_prod, tau, choices): 
        choices_one_hot = torch.empty((len(self.variables), self.supernet.cardinality), requires_grad=False)
        for variable in self.variables:
            neighbors = self.neighbors[variable]
            cardinality = len(neighbors)
            if cardinality == 1:
                index = torch.zeros(1)
            else:
                neighbor0 = choices[neighbors[0]]
                neighbor1 = choices[neighbors[1]]
                if cardinality == 2:
                    prob = torch.log(factor_prod[variable][:, neighbor0, neighbor1])
                    index = F.gumbel_softmax(prob, tau=tau, hard=True)
                else:
                    assert cardinality == 3
                    neighbor2 = choices[neighbors[2]]
                    prob = torch.log(factor_prod[variable][:, neighbor0, neighbor1, neighbor2])
                    index = F.gumbel_softmax(prob, tau=tau, hard=True)

            offset = torch.zeros((self.supernet.cardinality - index.shape[0]))
            choices[variable] = index.clone().detach().argmax()
            choices_one_hot[variable] = torch.cat((index, offset))
        
        return choices_one_hot


    def get_choices_one_hot(self, mode='none'):
        choices_one_hot = torch.zeros((len(self.variables), self.supernet.cardinality), requires_grad=False)
        
        channel_step = self.supernet.channel_step
        for variable in self.variables:
            cards = self.num_states[variable]
            if cards == 1:
                index = 0
            else:
                if mode == 'max5':
                    index = cards - 1
                elif mode == 'max3':
                    index = channel_step - 1
                else:
                    prob = torch.ones(cards)
                    index = torch.multinomial(prob, 1)
            choices_one_hot[variable][index] = 1
        
        return choices_one_hot


    def encoding_to_one_hot(self, choices_encoding):
        choices_one_hot = torch.zeros((len(self.variables), self.supernet.cardinality), requires_grad=False)
        for variable in self.variables:
            index = choices_encoding[variable]
            choices_one_hot[variable][index] = 1
        
        return choices_one_hot


class SuperNet(nn.Module):
    def __init__(self, image_channels, num_classes, channel_step=5):
        super(SuperNet, self).__init__()
        
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.channel_step = channel_step

        def divise8(i):
            return (np.maximum(np.round(i / 8), 1) * 8).astype(int)

        base_channels = np.array([32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512,
                                  256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32])
        self.configurations = divise8(base_channels.reshape(-1, 1) * np.linspace(0.5, 1.5, self.channel_step).reshape(1, -1))     
        channels = [self.image_channels] + [int(c[-1]) for c in self.configurations]
        self.layers = len(channels)

        self.kernel_list = [3, 5]
        self.kernel_step = len(self.kernel_list)
        self.cardinality = self.channel_step * self.kernel_step
        
        self.down_idx = [2, 5, 8, 11]
        self.up_idx = [14, 17, 20, 23]
        self.step = len(self.down_idx)

        self.block = nn.ModuleList([])

        for i in range(self.layers - 1):
            block = nn.ModuleList([])
            if i in self.down_idx:
                block.append(Down(channels[i], channels[i+1], supernet=True))
            elif i in self.up_idx:
                block.append(Up(channels[i], channels[i+1], supernet=True))
            elif (i - 1) in self.up_idx:
                idx = self.step - self.up_idx.index(i-1) - 1
                in_channels = channels[i] + channels[self.down_idx[idx]]
                for kernel_size in self.kernel_list:
                    block.append(ChoiceBlock(in_channels, channels[i+1], kernel_size=kernel_size, stride=1, supernet=True))
            else:
                for kernel_size in self.kernel_list:
                    block.append(ChoiceBlock(channels[i], channels[i+1], kernel_size=kernel_size, stride=1, supernet=True))

            self.block.append(block)

        self.cat = Cat()
        self.filter = Filter()
        self.head = Head(channels[-1], self.num_classes)  
        
        
    def forward(self, x, choices_one_hot):
        step = self.step
        x_cp = [None] * step
        choices = torch.argmax(choices_one_hot, dim=1)
        kernel_choices = choices // self.channel_step
        channel_choices = choices % self.channel_step
        
        for i in range(0, self.up_idx[0]):
            if i in self.down_idx:
                j = 0
                x_cp[self.down_idx.index(i)] = x.clone()
            else:
                j = kernel_choices[i+1]

            for one_hot in choices_one_hot[i+1]:
                if one_hot:
                    k = channel_choices[i+1]
                    x = one_hot * self.block[i][j](x)
                    x = self.filter(x, self.configurations[i][k])
                    break

        step -= 1
        for i in range(self.up_idx[0], self.layers - 1):
            if i not in self.up_idx:
                for one_hot in choices_one_hot[i+1]:
                    if one_hot:
                        j = kernel_choices[i+1]
                        k = channel_choices[i+1]
                        x = one_hot * self.block[i][j](x)
                        x = self.filter(x, self.configurations[i][k])
                        break
            else:
                for one_hot in choices_one_hot[i+1]:
                    if one_hot:
                        j = 0
                        k = channel_choices[i+1]
                        x = one_hot * self.block[i][j](x)
                        x = self.filter(x, self.configurations[i][k])
                        x = self.cat(x, x_cp[step])
                        step -= 1
                        break
        
        return self.head(x)

 
class ChildNet(nn.Module):
    def __init__(self, image_channels, num_classes, channel_step, choices):
        super(ChildNet, self).__init__()

        assert len(choices) == 26 
        
        self.image_channels = image_channels
        self.num_classes = num_classes
        
        def divise8(i):
            return (np.maximum(np.round(i / 8), 1) * 8).astype(int)
        
        base_channels = np.array([32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512,
                                  256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32])
        configurations = divise8(base_channels.reshape(-1, 1) * np.linspace(0.5, 1.5, channel_step).reshape(1, -1))     
        kernel_choices = choices // channel_step
        channel_choices = choices % channel_step
        channels = [self.image_channels] + \
                    configurations[np.arange(configurations.shape[0]), channel_choices].tolist()
        self.layers = len(channels)
        
        self.kernel_list = [3, 5]
        
        self.down_idx = [2, 5, 8, 11]
        self.up_idx = [14, 17, 20, 23]
        self.step = len(self.down_idx)   
    
        self.block = nn.ModuleList([])
        
        for i in range(self.layers - 1):
            if i in self.down_idx:
                self.block.append(Down(channels[i], channels[i+1]))
            elif i in self.up_idx:
                self.block.append(Up(channels[i], channels[i+1]))
            elif (i - 1) in self.up_idx:
                kernel_size = self.kernel_list[kernel_choices[i]]
                idx = self.step - self.up_idx.index(i-1) - 1
                in_channels = channels[i] + channels[self.down_idx[idx]]
                self.block.append(ChoiceBlock(in_channels, channels[i+1], kernel_size=kernel_size))
            else:
                kernel_size = self.kernel_list[kernel_choices[i]]
                self.block.append(ChoiceBlock(channels[i], channels[i+1], kernel_size=kernel_size))

        self.cat = Cat()
        self.head = Head(channels[-1], self.num_classes)


    def forward(self, x):
        step = self.step
        x_cp = [None] * step
        for i in range(0, self.up_idx[0]):
            if i in self.down_idx:
                x_cp[self.down_idx.index(i)] = x.clone()
            x = self.block[i](x)
        
        step -= 1
        for i in range(self.up_idx[0], self.layers - 1):
            if i not in self.up_idx:
                x = self.block[i](x)
            else:
                x = self.block[i](x)
                x = self.cat(x, x_cp[step])
                step -= 1        
        
        return self.head(x)