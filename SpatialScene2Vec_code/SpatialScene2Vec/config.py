#path
from typing import Tuple


train_datafile = '/home/gesy/space2vec/spacegraph/data_collection/Place2Vec_center/neighborgraphs_training.pkl'
test_datafile = '/home/gesy/space2vec/spacegraph/data_collection/Place2Vec_center/neighborgraphs_test.pkl'
validation_datafile = '/home/gesy/space2vec/spacegraph/data_collection/Place2Vec_center/neighborgraphs_validation.pkl'

point_set_datafile = '/home/gesy/space2vec/spacegraph/data_collection/Place2Vec_center/pointset.pkl'
model_dir = './model/'
log_dir = './log/'

#parameters
num_context_sample = 3
embed_dim = 64
dropout = 0.5
enc_agg = ['mean','min','max']
model_type = ['relative','global','join','together']

#RBF
num_rbf_anchor_pts = 0
rbf_kernal_size = 10e2
rbf_kernal_size_ratio = 0

#Space Encoder
spa_enc = ['gridcell','hexagridcell','theory']
spa_embed_dim = 64
freq = 32
max_radius = 40000
min_radius = 50
spa_f_act = ['sigmoid','relu']
freq_init = 'geometric'
spa_enc_use_layn = 'T'#whether to use layer normalization in spa_enc
spa_enc_use_postmat = 'T'#whether to use post martix in spa_enc

#global space/position encoder
g_spa_enc = ['gridcell','hexagridcell','theory']
g_spa_embed_dim = 64
g_freq = 32
g_max_radius = 40000
g_min_radius = 50
g_spa_f_act = ['sigmoid','relu']
g_freq_init = 'geometric'
g_spa_enc_use_layn = 'T'#whether to use layer normalization in spa_enc
g_spa_enc_use_postmat = 'T'#whether to use post martix in spa_enc


#ffn
num_hidden_layer = 1
hidden_dim = 512
use_layn = 'T'
skip_connection = 'T'



use_dec = 'T'#whether to use another decoder following the initial decoder
#initial decoder,without query embedding
init_decoder_atten_type = ['concat','g_pos_concat']
init_decoder_atten_act = 'leakyrelu'
init_decoder_atten_f_act = 'sigmoid'
init_decoder_atten_num = 1
init_decoder_use_layn = 'T'
init_decoder_use_postmat = 'T'

#decoder
decoder_atten_type = ['concat','g_pos_concat']
decoder_atten_act = 'leakyrelu'
decoder_atten_f_act = 'sigmoid'
decoder_atten_num = 1
decoder_use_layn = 'T'
decoder_use_postmat = 'T'

#encoder decoder
join_dec_type = ['max','min','mean','cat']
act = ['sigmoid','relu']

#train
opt = ['adam','sgd']
lr = 0.001
max_iter = 2500
max_burn_in = 5000#the maximum iterator for relative/global model converge
batch_size = 256
tol = 0.000001

#eval
log_every = 50
val_every = 50

#cuda
cuda = True

