from turtle import forward
import torch
import torch.nn as nn
from scene2vec_code.module import get_activation_function

import torch
import torch.nn as nn
from scene2vec_code.module import get_activation_function,MultiLayerFeedForwardNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = ('cpu')
class Scene2Vec(nn.Module):
    '''
    Build Scene2Vec model
    '''
    def __init__(self,
                 poiset,
                 centerset,
                 position_encoder,
                 feature_encoder,
                 activate='relu',
                 num_poi_sample=10,
                 num_neg_sample=10):
        super(Scene2Vec,self).__init__()
        self.poiset = poiset
        self.centerset = centerset
        self.position_encoder = position_encoder
        self.feature_encoder = feature_encoder
        self.num_poi_sample = num_poi_sample
        self.activation = get_activation_function(activation=activate,context_str='Scene2Vec')
        self.num_neg_sample = num_neg_sample
        self.linear = MultiLayerFeedForwardNN(
            input_dim=self.num_poi_sample*64,
            output_dim=256,
            num_hidden_layers=1,
            dropout_rate=0.3,
            hidden_dim=2048,
            use_layernormalize=True,
            skip_connection=True,
            context_str='Scene2Vec_linear')

    def sample_poi(self,ng_list):
        for ng in ng_list:
            ng.sample_poi_list(self.num_poi_sample)
    
    def sample_neg_pts(self,ng_list):
        for ng in ng_list:
            ng.sample_neg(self.num_neg_sample)

    def forward(self,ng_list,do_full_eval=False):
        if not do_full_eval:
            self.sample_neg_pts(ng_list)
        coords = self.get_spa_coords(ng_list)
        spa_embeds = self.position_encoder(coords)
        fea_embed = self.get_feature_embed(ng_list)
        scene_embed = spa_embeds+fea_embed
        scene_embed = scene_embed.view(len(ng_list),-1)
        scene_embed = self.linear(scene_embed)
        return scene_embed


    def get_center_pt_embed(self, ng_list):
        '''
        Given a list of NeighborGraph(), get the feature embedding of the center points
        Return:
            query_embed: shape (batch_size, embed_dim)
        '''
        pt_list = [ng.center_pt for ng in ng_list]

        # query_embed: shape (batch_size, embed_dim)
        query_embed = self.enc(pt_list)
        return query_embed

    def get_spa_coords(self, ng_list):
        '''
        Given a list of NeighborGraph(), get their (deltaX, deltaY) list
        '''
        coords = []
        for ng in ng_list:
            cur_coords = []
            center_coord = self.centerset.pt_dict[ng.center_pt].coord
            for i in range(len(ng.poi_list)):
                coord = self.poiset.pt_dict[ng.poi_list[i]].coord
                cur_coords.append([coord[0]-center_coord[0], coord[1]-center_coord[1]])
            coords.append(cur_coords)
        # coords: shape (batch_size, num_context_sample, 2)
        return coords
    
    def get_feature_embed(self,ng_list):
        pt_list = []
        for ng in ng_list:
            pt_list += list(ng.poi_list)
        feature_embed = self.feature_encoder(pt_list)
        # print(len(pt_list),feature_embed.shape)
        feature_embed = feature_embed.view(len(ng_list),self.num_poi_sample,-1) ###关键点
        return feature_embed

    # def get_neg_pt_embed(self,ng_list,do_full_eval = True):
    #     '''
    #     Given a list of NeighborGraph(), get the scene embedding of the negative samples
    #     Return:
    #         key_embeds: shape (batch_size, num_neg_sample, embed_dim)
    #     '''
    #     if do_full_eval == True:
    #         num_neg_sample = len(ng_list[0].neg_sample)
    #         pt_list = []
    #         for ng in ng_list:
    #             pt_list += list(ng.neg_sample)

    #         # key_embeds: shape (batch_size*num_neg_sample, embed_dim)
    #         key_embeds = self.enc(pt_list)
    #         # key_embeds: shape (batch_size, num_neg_sample, embed_dim)
    #         key_embeds = key_embeds.view(len(ng_list), num_neg_sample, -1)
    #     else:
    #         # pt_list: shape (batch_size*num_neg_resample)
    #         pt_list = []
    #         for ng in ng_list:
    #             pt_list += list(ng.sample_neg_poi)

    #         # key_embeds: shape (batch_size*num_neg_resample, embed_dim)
    #         key_embeds = self.enc(pt_list)
    #         # key_embeds: shape (batch_size, num_neg_resample, embed_dim)
    #         key_embeds = key_embeds.view(len(ng_list), self.num_neg_sample, -1)
    #     return key_embeds

class ConstraintModel(nn.Module):
    def __init__(self,encoder_q,encoder_k,T=0.07,m=0.999):
        super(ConstraintModel,self).__init__()
        self.T = T
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.m = m

        for param_q,param_k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            if not param_q.requires_grad:
                param_q.requires_grad = True
            # print(param_q.shape,param_q.requires_grad,param_k.requires_grad)
        
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size
        labels = torch.arange(N, dtype=torch.long).to(device)
        acc = self.accuracy(logits,labels,topk=(1,3,5,10))
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T),acc

    def forward(self,ng_list_q,ng_list_k):
        q1 = self.encoder_q(ng_list_q)
        q2 = self.encoder_q(ng_list_k)
        with torch.no_grad():
            self._update_momentum_encoder()
            k1 = self.encoder_k(ng_list_q)
            k2 = self.encoder_k(ng_list_k)
        loss_acc_1 = self.contrastive_loss(q1,k2)
        loss_acc_2 = self.contrastive_loss(q2,k1)
        loss = loss_acc_1[0] + loss_acc_2[0]
        # print(loss_acc_1[1],loss_acc_2[1])
        # print('--------------------------------')
        acc = list(map(lambda x:(x[0]+x[1])/2,zip(loss_acc_1[1],loss_acc_2[1])))
        return loss,acc

    def accuracy(self,output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            # print(batch_size)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            # print(correct_k)
            return res

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Momentum update of the momentum encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
