# import pickle
import torch
import numpy as np
from collections import defaultdict
# def _random_sampling(item_tuple, num_sample):
#     '''
#     poi_type_tuple: (Type1, Type2,...TypeM)
#     '''

#     type_list = list(item_tuple)
#     if len(type_list) > num_sample:
#         return tuple(np.random.choice(type_list, num_sample, replace=False))
#     elif len(type_list) == num_sample:
#         return item_tuple
#     else:
#         return tuple(np.random.choice(type_list, num_sample, replace=True))

class POI():
    def __init__(self,id,coord,feature,data_mode,feature_func=None,num_sample=None):
        self.id = id
        self.coord = tuple([coord[i] for i in range(len(coord))])
        self.coord_dim = len(coord)
        if feature_func is not None and num_sample is not None:
            self.features = feature_func(feature,num_sample)
        else:
            self.features = feature
        self.data_mode = data_mode

    def __hash__(self) -> int:
        return hash((self.id,self.coord,self.features))
    
    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __neq__(self,other):
        return self.id != other.id

    def __str__(self) -> str:
        return "{}:coord: ({}) features: ({})".format(self.id," ".join(list(self.coord))," ".join(list(self.features)))
    
    def serialize(self):
        return (self.id,self.coord,self.features,self.data_mode)

class POI_Datasets():
    def __init__(self,poi_list,feature_embed_lookup,feature_dim,feature_mode,data_mode):
        self.feature_embed_lookup = feature_embed_lookup
        self.feature_dim = feature_dim
        self.feature_mode = feature_mode

        self.pt_dict = defaultdict()
        self.pt_mode = defaultdict()
        self.pt_mode['training'] = set()
        self.pt_mode['validation'] = set()
        self.pt_mode['test'] = set()
        self.pt_mode['POI'] = set()
        for poi_tuple in poi_list:
            id,feature,coord = poi_tuple
            self.pt_dict[id] = POI(id=id,coord=coord,feature=feature,data_mode=data_mode,feature_func=None,num_sample=None)
            self.pt_mode[data_mode].add(id)

    def serialize(self):
        '''
        Serialize the pointset
        '''
        pt_list = []
        for id in self.pt_dict:
            pt_list.append(self.pt_dict[id].serialize())

        return (self.num_feature_type, pt_list)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
def load_poiset(poi_list,embed_dim = 64):
    # poi_list = pickle.load(open(data_path,'rb'))
    num_poi_type = 22
    feature_dim = embed_dim
    feature_embedding = torch.nn.Embedding(num_embeddings=num_poi_type,embedding_dim=embed_dim)
    feature_embedding.weight.data.normal_(0,1./embed_dim)

    feature_embed_lookup = lambda pt_types:feature_embedding(
        torch.autograd.Variable(torch.LongTensor(pt_types).to(device))
    )
    poidataset = POI_Datasets(poi_list=poi_list,feature_embed_lookup=feature_embed_lookup,feature_dim=feature_dim,feature_mode="TYPE",data_mode='POI')
    return poidataset,feature_embedding
