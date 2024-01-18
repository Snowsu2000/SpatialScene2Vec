# import pickle
import numpy as np

def _random_sampling(item_tuple, num_sample):
    '''
    poi_type_tuple: (Type1, Type2,...TypeM)
    '''

    type_list = list(item_tuple)
    if len(type_list) > num_sample:
        return tuple(np.random.choice(type_list, num_sample, replace=False))
    elif len(type_list) == num_sample:
        return item_tuple
    else:
        return tuple(np.random.choice(type_list, num_sample, replace=True))
class Scene_Datasets():
    def __init__(self,scene_list) -> None:
        '''
        Load scene_list
        '''
        self.center_pt = scene_list[0]
        self.poi_list = tuple(scene_list[1])
        self.neg_sample = scene_list[2]
        self.sample_poi_list = None
        self.sample_neg_poi = None

    def sample_poi_list(self,num_sample):
        self.sample_poi_list = _random_sampling(self.context_pts, num_sample)
    
    def sample_neg(self,num_neg_sample):
        self.sample_neg_poi = list(_random_sampling(self.neg_sample,num_neg_sample))
    
    def __hash__(self):
        return hash(self.center_pt,self.poi_list,self.neg_sample)

    def __eq__(self, other):
        return (self.center_pt,self.poi_list,self.neg_sample) == (other.center_pt,other.poi_list,other.neg_sample)
    
    def __neq__(self,other):
        return self.__hash__() != other.__hash__()
    
    def serialize(self):
        return (self.center_pt,self.poi_list,self.neg_sample)
    
    @staticmethod
    def deserialize(serial_info):
        return Scene_Datasets(serial_info)
    
def load_scene_list(ng_data):#data_file
    # raw_info = pickle.load(open(data_file,'rb'))
    raw_info = ng_data
    ng_list = []
    for info in raw_info:
        info_1 = Scene_Datasets.deserialize(info[0])
        info_2 = Scene_Datasets.deserialize(info[1])
        ng_list.append([info_1,info_2])
    return ng_list