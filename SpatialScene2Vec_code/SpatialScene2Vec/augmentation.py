from distutils.dir_util import copy_tree
import random
from pyproj import CRS
from pyproj import Transformer
import numpy as np
import copy
class SpatialSceneAugmentation:
    def __init__(self,poiset,centerset,max_poi_len=10) -> None:
        self.poiset = copy.deepcopy(poiset)
        self.centerset = centerset
        self.max_poi_len = max_poi_len
        # self.shift = 0.0001
        self.p = 0.8
        self.crs_org = CRS.from_epsg(4326)
        self.crs_convert = CRS.from_epsg(3857)
        self.transformer = Transformer.from_crs(self.crs_org,self.crs_convert)
        self.transformer_reverse = Transformer.from_crs(self.crs_convert,self.crs_org)
        # self.scene_add_list =  [3,6,7,9,16,17]
        self.poi_len = len(self.poiset)
    
    def __call__(self, scnen_list):
        scene_list_pair = []
        for scene in scnen_list:
            scene_len = len(scene[1])
            org_scene = copy.deepcopy(scene)
            #1.delete the scenes with 0 poi
            if scene_len == 0:
                continue
            #2.if the num of pois in scene is less than 5
            #  add 1-3 pois randomly
            if scene_len < 5:
                sim_scene,new_poiset = self.add_or_delete(scene,is_add=True)
                self.poiset += new_poiset
            else:
                sim_scene = self.add_or_delete(scene,is_add=False)
            sim_scene,new_poiset = self.shifting(sim_scene)
            self.poiset += new_poiset
            org_scene,new_poiset = self.padding(org_scene)
            self.poiset += new_poiset
            sim_scene,new_poiset = self.padding(sim_scene)
            self.poiset += new_poiset
            scene_list_pair.append([org_scene,sim_scene])
        return scene_list_pair,self.poiset




    def add_or_delete(self,scene,is_add=True):
        if is_add:
            num = random.randint(1,2)
            center_x,center_y = self.transformer_reverse.transform(self.centerset[scene[0]][1][0],self.centerset[scene[0]][1][1])
            new_pois_num = []
            new_poiset = []
            for i in range(num):
                new_poi_loc = self.transformer.transform((random.random()*2-1)*0.002+center_x,
                                               (random.random()*2-1)*0.002+center_y)
                new_poi_type = random.randint(0,21)
                new_pois_num.append(self.poi_len+i)
                new_poiset.append([self.poi_len+i,new_poi_type,new_poi_loc])
            sim_scene = [scene[0],scene[1]+new_pois_num,scene[2]]
            self.poi_len += num
            return sim_scene,new_poiset
        else:
            num = random.randint(0,2)
            new_poi_list = list(np.random.choice(scene[1],len(scene[1])-num,replace=False))
            sim_scene = [scene[0],new_poi_list,scene[2]]
            return sim_scene

    #后面考虑概率进行平移
    def shifting(self,scene):
        num = random.randint(0,len(scene[1])-1)
        new_poiset = []
        if num == 0:
            return scene,new_poiset
        else:
            shift_poi_list = list(np.random.choice(scene[1],num,replace=False))
            new_poi_list = []
            for poi_id in scene[1]:
                if poi_id in shift_poi_list:
                    continue
                new_poi_list.append(poi_id)
            for i,poi_id in enumerate(shift_poi_list):
                org_poi_x,org_poi_y = self.transformer_reverse.transform(self.poiset[poi_id][2][0],self.poiset[poi_id][2][1])
                shift_poi_loc = self.transformer.transform((random.random()*2-1)*0.0005+org_poi_x,
                                                            (random.random()*2-1)*0.0005+org_poi_y)
                new_poi_list.append(self.poi_len+i)
                new_poiset.append([self.poi_len+i,self.poiset[poi_id][1],shift_poi_loc])
            self.poi_len += num
            sim_scene = [scene[0],new_poi_list,scene[2]]
            return sim_scene,new_poiset             

    def padding(self,scene):
        new_poiset = []
        padding_len = self.max_poi_len - len(scene[1])
        new_poi_list = copy.deepcopy(scene[1])
        if padding_len == 0:
            return scene,new_poiset
        else:
            for i in range(padding_len):
                poi_choose = int(np.random.choice(scene[1],1,replace=False)[0])
                org_poi_x,org_poi_y = self.transformer_reverse.transform(self.poiset[poi_choose][2][0],self.poiset[poi_choose][2][1])
                padding_poi_loc = self.transformer.transform((random.random()*2-1)*0.0003+org_poi_x,
                                                            (random.random()*2-1)*0.0003+org_poi_y)
                new_poi_list.append(self.poi_len+i)
                new_poiset.append([self.poi_len+i,21,padding_poi_loc])
            padding_scene = [scene[0],new_poi_list,scene[2]]
            self.poi_len += padding_len
            return padding_scene,new_poiset