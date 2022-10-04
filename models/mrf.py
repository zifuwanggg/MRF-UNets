import os
import pickle

import torch
import numpy as np
from thop import profile
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor

from models.ops import Up, Down, Head, ChoiceBlock


class MRF(): 
    def __init__(self, supernet, potentials, gamma, lams=[], solutions=[], flops_path="flops.pkl"):
        assert len(lams) == len(solutions)
        
        self.supernet = supernet
        
        self.mrf = self.create_mrf()
        
        unaries = potentials[0 : self.supernet.layers + 1]
        pairwises = potentials[self.supernet.layers + 1 : ]
        self.unary_loss = self.create_unary_potentials(unaries)
        self.pairwise_loss = self.create_pairwise_potentials(pairwises)

        self.unary_dis = self.create_unary(ones=False)
        if len(lams) > 0 and len(solutions) > 0:
            for lam, solution in zip(lams, solutions):
                self.unary_dis = self.add_dissimilarity(self.unary_dis, lam, solution)
                self.unary_loss = self.create_combined_unary_potentials() 

        if os.path.isfile(flops_path):
            with open(flops_path, 'rb') as h: 
                self.pairwise_flops = pickle.load(h)
        else:
            self.pairwise_flops = self.create_pairwise_flops()
            with open(flops_path, 'wb') as f:
                pickle.dump(self.pairwise_flops, f, protocol=4)
  
        self.unary = self.unary_loss
        self.pairwise = self.create_combined_pairwise_potentials(gamma)
        
        self.create_factor(self.unary)
        self.create_factor(self.pairwise)


    def create_mrf(self):
        edges = []
        for i in range(self.supernet.layers):
            edges.append((i, i+1))

        down_idx = self.supernet.down_idx
        up_idx = [x + 2 for x in self.supernet.up_idx]
        up_idx.reverse()

        for down, up in zip(down_idx, up_idx):
            edges.append((down, up))

        mrf = MarkovModel()
        mrf.add_edges_from(edges)
        
        return mrf


    def get_choice_encoding(self, choices):
        weights = np.array([self.supernet.channel_step, self.supernet.channel_step, 1])
        choice_encoding = np.dot(choices, weights)
        choice_encoding = np.insert(choice_encoding, 0, values=0)
        
        return np.append(choice_encoding, values=0)

    
    def get_flops(self, choice_encoding):
        flops = 0
        for l in range(self.supernet.layers):    
            flops += self.pairwise_flops[(l, l+1)][choice_encoding[l], choice_encoding[l+1]]
            if (l - 2) in self.supernet.up_idx:
                idx = self.supernet.step - self.supernet.up_idx.index(l-2) - 1
                idx = self.supernet.down_idx[idx]
                flops += self.pairwise_flops[(idx, l)][choice_encoding[idx], choice_encoding[l]] 
        
        return flops


    def get_score(self, choice_encoding, pairwise=True):
        score = 0

        for l in range(self.supernet.layers + 1):
            score += self.unary[l][choice_encoding[l]]

        if pairwise:
            for l in range(self.supernet.layers):    
                score += self.pairwise[(l, l+1)][choice_encoding[l], choice_encoding[l+1]]
                if (l - 2) in self.supernet.up_idx:
                    idx = self.supernet.step - self.supernet.up_idx.index(l-2) - 1
                    idx = self.supernet.down_idx[idx]
                    score += self.pairwise[(idx, l)][choice_encoding[idx], choice_encoding[l]] 
        
        return score

    
    def create_unary(self, ones):
        unary = {}

        for l in range(self.supernet.layers + 1):
            if l in [0, self.supernet.layers]:
                if ones: unary[l] = np.ones(1)
                else: unary[l] = np.zeros(1)

            elif (l - 1) in (self.supernet.down_idx + self.supernet.up_idx):
                if ones: unary[l] = np.ones(self.supernet.channel_step)
                else: unary[l] = np.zeros(self.supernet.channel_step) 

            else:
                if ones: unary[l] = np.ones(self.supernet.cardinality)    
                else: unary[l] = np.zeros(self.supernet.cardinality)
        
        return unary


    def create_unary_loss(self, samples, alpha):
        unary = self.create_unary(ones=False)
        unary[0] = np.ones(1)
        unary[self.supernet.layers] = np.ones(1)
        
        for key, values in samples.items():
            for value in values:
                choices = np.array(key).reshape(3, self.supernet.layers - 1).T
                choice_encoding = self.get_choice_encoding(choices)[1:-1]
                for l in range(0, self.supernet.layers - 1):
                    unary[l+1][choice_encoding[l]] = (1 - alpha) * unary[l+1][choice_encoding[l]] \
                        + alpha * value   
        
        return unary


    def create_unary_potentials(self, unaries):
        unary = self.create_unary(ones=True)
        for item in unaries:
            l = int(item[0].split('.')[1])
            unary[l] = np.exp(item[1].detach().numpy())
        
        return unary


    def add_dissimilarity(self, unary, lam, solution):  
        for l in range(1, self.supernet.layers):
            unary[l][solution[l]] -= (self.unary_loss[l].max() - self.unary_loss[l].min()) / lam
        
        return unary

    
    def create_combined_unary_potentials(self):
        unary = self.create_unary(ones=False)
        for key in unary.keys():
            unary[key] = self.unary_loss[key] + self.unary_dis[key]
            assert unary[key].min() > 0
        
        return unary


    def create_pairwise(self, array=True):
        pairwise = {}
    
        for l in range(self.supernet.layers):    
            if l == 0:
                if array:
                    pairwise[(l, l+1)] = np.zeros((1, self.supernet.cardinality))
                else:
                    pairwise[(l, l+1)] = [[0 for i in range(self.supernet.cardinality)] for j in range(1)]
            elif l == self.supernet.layers - 1:
                if array:
                    pairwise[(l, l+1)] = np.zeros((self.supernet.cardinality, 1))
                else:
                    pairwise[(l, l+1)] = [[0 for i in range(1)] for j in range(self.supernet.cardinality)]
            elif l in self.supernet.down_idx:
                if array:
                    pairwise[(l, l+1)] = np.zeros((self.supernet.cardinality, self.supernet.channel_step))
                else:
                    pairwise[(l, l+1)] = [[0 for i in range(self.supernet.channel_step)] for j in range(self.supernet.cardinality)]
            elif (l - 1) in self.supernet.down_idx:
                if array:
                    pairwise[(l, l+1)] = np.zeros((self.supernet.channel_step, self.supernet.cardinality))
                else:
                    pairwise[(l, l+1)] = [[0 for i in range(self.supernet.cardinality)] for j in range(self.supernet.channel_step)]
            elif l in self.supernet.up_idx:
                if array:
                    pairwise[(l, l+1)] = np.zeros((self.supernet.cardinality, self.supernet.channel_step))
                else:
                    pairwise[(l, l+1)] = [[0 for i in range(self.supernet.channel_step)] for j in range(self.supernet.cardinality)]
            elif (l - 1) in self.supernet.up_idx:
                if array:
                    pairwise[(l, l+1)] = np.zeros((self.supernet.channel_step, self.supernet.cardinality))
                else:
                    pairwise[(l, l+1)] = [[0 for i in range(self.supernet.cardinality)] for j in range(self.supernet.channel_step)]
            else:
                if array:
                    pairwise[(l, l+1)] = np.zeros((self.supernet.cardinality, self.supernet.cardinality))
                else:
                    pairwise[(l, l+1)] = [[0 for i in range(self.supernet.cardinality)] for j in range(self.supernet.cardinality)]
            if (l - 2) in self.supernet.up_idx:
                idx = self.supernet.step - self.supernet.up_idx.index(l-2) - 1
                idx = self.supernet.down_idx[idx]
                if array:
                    pairwise[(idx, l)] = np.zeros((self.supernet.cardinality, self.supernet.cardinality))
                else:
                    pairwise[(idx, l)] = [[0 for i in range(self.supernet.cardinality)] for j in range(self.supernet.cardinality)]
        
        return pairwise

    
    def create_pairwise_potentials(self, pairwises):
        pairwise = self.create_pairwise()
        for item in pairwises:
            edge = item[0].split('.')[1].split(',')
            s, t = int(edge[0]), int(edge[1])
            pairwise[(s, t)] = np.exp(item[1].detach().numpy())
        
        return pairwise
    

    def create_pairwise_flops(self):
        pairwise = self.create_pairwise()
        
        input_size = 256
        for l in range(self.supernet.layers):
            if l == 0:
                x = i = 0
                in_channels = self.supernet.image_channels
                inputs = torch.randn(1, in_channels, input_size, input_size)
                for j, out_channels in enumerate(self.supernet.configurations[l]):
                    for k_out, kernel_size in enumerate(self.supernet.kernel_list):
                        y = self.supernet.channel_step * k_out + j
                        cb = ChoiceBlock(in_channels, out_channels, kernel_size)
                        pairwise[(l, l+1)][x, y] = profile(cb, inputs=(inputs, ), verbose=False)[0] / 1e9
            elif l == self.supernet.layers - 1:
                y = j = 0
                out_channels = self.supernet.num_classes
                for i, in_channels in enumerate(self.supernet.configurations[l-1]):
                    for k_in, kernel_size in enumerate(self.supernet.kernel_list):
                        x = self.supernet.channel_step * k_in + i
                        inputs = torch.randn(1, in_channels, input_size, input_size)
                        cb = Head(in_channels, out_channels)
                        pairwise[(l, l+1)][x, y] = profile(cb, inputs=(inputs, ), verbose=False)[0] / 1e9
            elif l in self.supernet.down_idx:
                for i, in_channels in enumerate(self.supernet.configurations[l-1]):
                    inputs = torch.randn(1, in_channels, input_size, input_size)
                    for j, out_channels in enumerate(self.supernet.configurations[l]):
                        for k_in, kernel_size in enumerate(self.supernet.kernel_list): 
                            x = self.supernet.channel_step * k_in + i
                            y = j
                            cb = Down(in_channels, out_channels)
                            pairwise[(l, l+1)][x, y] = profile(cb, inputs=(inputs, ), verbose=False)[0] / 1e9
                input_size = int(input_size / 2)
            elif (l - 1) in self.supernet.up_idx + self.supernet.down_idx:
                for i, in_channels in enumerate(self.supernet.configurations[l-1]):
                    inputs = torch.randn(1, in_channels, input_size, input_size)
                    for j, out_channels in enumerate(self.supernet.configurations[l]):
                        for k_out, kernel_size in enumerate(self.supernet.kernel_list): 
                            x = i
                            y = self.supernet.channel_step * k_out + j
                            cb = ChoiceBlock(in_channels, out_channels, kernel_size)
                            pairwise[(l, l+1)][x, y] = profile(cb, inputs=(inputs, ), verbose=False)[0] / 1e9
            elif l in self.supernet.up_idx:
                for i, in_channels in enumerate(self.supernet.configurations[l-1]):
                    inputs = torch.randn(1, in_channels, input_size, input_size)
                    for j, out_channels in enumerate(self.supernet.configurations[l]):
                        for k_in, kernel_size in enumerate(self.supernet.kernel_list): 
                            x = self.supernet.channel_step * k_in + i
                            y = j
                            cb = Up(in_channels, out_channels)
                            pairwise[(l, l+1)][x, y] = profile(cb, inputs=(inputs, ), verbose=False)[0] / 1e9
                input_size = int(input_size * 2)
            else:
                for i, in_channels in enumerate(self.supernet.configurations[l-1]):
                    inputs = torch.randn(1, in_channels, input_size, input_size)
                    for j, out_channels in enumerate(self.supernet.configurations[l]):
                        for k_in, kernel_size_in in enumerate(self.supernet.kernel_list): 
                            for k_out, kernel_size_out in enumerate(self.supernet.kernel_list):
                                x = self.supernet.channel_step * k_in + i
                                y = self.supernet.channel_step * k_out + j
                                cb = ChoiceBlock(in_channels, out_channels, kernel_size_out)
                                pairwise[(l, l+1)][x, y] = profile(cb, inputs=(inputs, ), verbose=False)[0] / 1e9
            if (l - 2) in self.supernet.up_idx:
                idx = self.supernet.step - self.supernet.up_idx.index(l-2) - 1
                idx = self.supernet.down_idx[idx]
                for i, in_channels in enumerate(self.supernet.configurations[idx-1]):
                    inputs = torch.randn(1, in_channels, input_size, input_size)
                    for j, out_channels in enumerate(self.supernet.configurations[l]):
                        for k_in, kernel_size_in in enumerate(self.supernet.kernel_list): 
                            for k_out, kernel_size_out in enumerate(self.supernet.kernel_list):
                                x = self.supernet.channel_step * k_in + i
                                y = self.supernet.channel_step * k_out + j
                                cb = ChoiceBlock(in_channels, out_channels, kernel_size_out)
                                pairwise[(idx, l)][x, y] = profile(cb, inputs=(inputs, ), verbose=False)[0] / 1e9
        
        return pairwise
    

    def create_combined_pairwise_potentials(self, gamma):
        pairwise = self.create_pairwise()
        for key in pairwise.keys():
            pairwise[key] = self.pairwise_loss[key] * np.exp(-gamma * self.pairwise_flops[key])
        
        return pairwise

    
    def create_factor(self, dic):
        for (key, value) in dic.items():
            key = [key] if type(key) == int else list(key)
            factor = DiscreteFactor(key, value.shape, value)
            self.mrf.add_factors(factor)