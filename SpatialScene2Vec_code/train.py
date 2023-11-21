import os
import pickle
from scene2vec_code.scene_data_load import load_scene_list
from scene2vec_code.poi_data_load import load_poiset
from scene2vec_code.center_data_load import load_centerset
from scene2vec_code.augmentation import SpatialSceneAugmentation
from scene2vec_code.Trainer import Trainer
path = os.getcwd()
data_path = '/data_in_use/'
scene_data = pickle.load(open(path+data_path+'scene_ng_list.pkl','rb'))
poiset = pickle.load(open(path+data_path+'pointset.pkl','rb'))
centerset = pickle.load(open(path+data_path+'centersets.pkl','rb'))
#Initialize the aumentation methon
Scene_aug = SpatialSceneAugmentation(poiset=poiset,centerset=centerset,max_poi_len=10)
scene_ng_pair,new_poiset = Scene_aug(scene_data)
scene_ng_list = load_scene_list(ng_data=scene_ng_pair)
poi_ng_list,feature_embedding = load_poiset(poi_list=new_poiset)
center_ng_list = load_centerset(center_list=centerset)
trainer = Trainer(poiset=poi_ng_list,
                  centerset=center_ng_list,
                  ng_pair_list=scene_ng_list,
                  feature_embedding=feature_embedding,
                  console=True)
trainer.train()