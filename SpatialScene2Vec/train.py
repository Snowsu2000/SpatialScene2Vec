#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
from scene2vec_code.scene_data_load import load_scene_list
from scene2vec_code.poi_data_load import load_poiset
from scene2vec_code.center_data_load import load_centerset


# ### 读取原始数据

# In[2]:



path = os.getcwd()
data_path = '/data_aug/'
scene_data = pickle.load(open(path+data_path+'scene_ng_list.pkl','rb'))
poiset = pickle.load(open(path+data_path+'pointset.pkl','rb'))
centerset = pickle.load(open(path+data_path+'centersets.pkl','rb'))


# ### 数据增强
# 增删、平移获得相似数据

# In[3]:


from scene2vec_code.augmentation import NewSpatialSceneAugmentation as spaug


# In[4]:


#Initialize the aumentation methon
Scene_aug = spaug(poiset=poiset,centerset=centerset,max_poi_len=10)


# In[5]:


scene_ng_pair,new_poiset = Scene_aug(scene_data)


# ### 数据格式整理

# In[6]:


scene_ng_list = load_scene_list(ng_data=scene_ng_pair)
poi_ng_list,feature_embedding = load_poiset(poi_list=new_poiset)
center_ng_list = load_centerset(center_list=centerset)


# ### 加载训练器

# In[7]:


from scene2vec_code.Trainer import Trainer


# In[8]:


trainer = Trainer(poiset=poi_ng_list,
                  centerset=center_ng_list,
                  ng_pair_list=scene_ng_list,
                  feature_embedding=feature_embedding,
                  console=True)


# In[9]:


trainer.train()


# In[ ]:




