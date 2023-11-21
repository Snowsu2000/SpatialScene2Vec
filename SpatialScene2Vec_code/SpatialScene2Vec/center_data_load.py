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

class Center():
    '''
    中心点集信息获取
    id:序号
    coord:墨卡托投影后的经纬度信息
    '''
    def __init__(self,id,coord,data_mode):
        self.id = id
        self.coord = tuple([coord[i] for i in range(len(coord))])
        self.coord_dim = len(coord)
        self.data_mode = data_mode

    def __hash__(self) -> int:
        return hash((self.id,self.coord))
    
    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __neq__(self,other):
        return self.id != other.id

    def __str__(self) -> str:
        return "{}:coord: ({})".format(self.id," ".join(list(self.coord)))
    
    def serialize(self):
        return (self.id,self.coord,self.data_mode)

class CenterDatasets():
    def __init__(self,poi_list,data_mode):

        self.pt_dict = defaultdict()
        self.pt_mode = defaultdict()
        self.pt_mode['training'] = set()
        self.pt_mode['validation'] = set()
        self.pt_mode['test'] = set()
        self.pt_mode['Center'] = set()
        for poi_tuple in poi_list:
            id,coord = poi_tuple
            self.pt_dict[id] = Center(id=id,coord=coord,data_mode=data_mode)
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
# device = ('cpu')
# device = torch.device("cpu")
def load_centerset(center_list):
    # center_list = pickle.load(open(data_path,'rb'))
    centerdataset = CenterDatasets(poi_list=center_list,data_mode='Center')
    return centerdataset
