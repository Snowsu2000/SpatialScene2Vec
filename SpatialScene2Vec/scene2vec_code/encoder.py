import torch
from torch.functional import norm
import torch.nn as nn
import numpy as np
import math
from scene2vec_code.module import *

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
          (frequency_num*1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = 1.0/timescales

    return freq_list

class RelationEncoder(nn.Module):
    def __init__(self,spa_embed_dim,coord_dim=2,frequency_num=16,max_radius=10000,min_radius=10,freq_init='geometric',ffn=None):
        super(RelationEncoder,self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))

    def cal_coord_embed(self,coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord,cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord,cur_freq)))
        return embed
    
    def cal_input_dim(self):
        return int(self.coord_dim*self.frequency_num*2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init,self.frequency_num,self.max_radius,self.min_radius)

    def cal_freq_mat(self):
        freq_mat = np.expand_dims(self.freq_list,axis=1)
        self.freq_mat = np.repeat(freq_mat,2,axis=1)
    
    def make_input_embeds(self,coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for RelationEncoder")
    
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis = 4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat
        
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))

        return spr_embeds
    
    def forward(self,coords):
        spr_embeds = self.make_input_embeds(coords)
        spr_embeds = torch.FloatTensor(spr_embeds)
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

class TheoryRelationEncoder(nn.Module):
    def __init__(self,spa_embed_dim,coord_dim=2,frequency_num=16,max_radius=10000,min_radius=1000,freq_init='geometric',ffn=None):
        super(TheoryRelationEncoder,self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn

        self.unit_vec1 = np.asarray([1.0, 0.0])                        # 0
        self.unit_vec2 = np.asarray([-1.0/2.0, math.sqrt(3)/2.0])      # 120 degree
        self.unit_vec3 = np.asarray([-1.0/2.0, -math.sqrt(3)/2.0])     # 240 degree


    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init,self.frequency_num)

    def cal_freq_mat(self):
        freq_mat = np.expand_dims(self.freq_list,axis=1)
        self.freq_mat = np.repeat(freq_mat,6,axis=1)

    def cal_input_dim(self):
        return int(6*self.frequency_num)

    def make_input_embeds(self,coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec 
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis = -1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis = -1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1
        
        return spr_embeds

    def forward(self,coords):
        spr_embed = self.make_input_embeds(coords)
        spr_embed = torch.FloatTensor(spr_embed)
        if self.ffn is not None:
            return self.ffn(spr_embed)
        else:
            return spr_embed

class PointFeatureEncoder(nn.Module):
    def __init__(self,feature_embed_lookup,feature_embedding,pointset,centerset,agg_func=torch.mean):
        super().__init__()
        self.feature_embed_lookup = feature_embed_lookup
        self.pointset = pointset
        self.centerset = centerset
        self.add_module('feature-embed',feature_embedding)
        self.agg_func = agg_func

    def forward(self,pt_list):
        feature_list = []
        # print([self.pointset.pt_dict[pt_list[0]].features],self.pointset.pt_dict[pt_list[0]].features)
        for pid in pt_list:
            feature_list.append([self.pointset.pt_dict[pid].features])
        
        embeds = self.feature_embed_lookup(feature_list)
        norm = embeds.norm(p=2,dim=2,keepdim=True)
        embeds_norm = embeds.div(norm.expand_as(embeds))
        aggs = self.agg_func(embeds_norm,dim=1,keepdim=False)
        # print(embeds.shape,norm.shape,embeds_norm.shape,aggs.shape)
        if type(aggs) == tuple:
            aggs = aggs[0]
        aggs_norm = aggs.norm(p=2,dim=1,keepdim=True)
        aggs_normalize = aggs.div(aggs_norm.expand_as(aggs))
        # print(len(pt_list),len(feature_list),aggs.shape,aggs_norm.shape,aggs_normalize.shape)
        return aggs_normalize