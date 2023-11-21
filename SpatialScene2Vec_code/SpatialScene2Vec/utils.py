# import torch
import logging
from scene2vec_code.encoder import *
import scene2vec_code.config as config
# from model import EncoderDecoder


def setup_console():
    logging.getLogger('').handlers = []
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def setup_logging(log_file,console=True,filemode='w'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode=filemode)
    if console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    return logging

def get_feature_encoder(feature_embed_lookup,feature_embedding,pointset,centerset):
    enc = PointFeatureEncoder(feature_embed_lookup=feature_embed_lookup,feature_embedding=feature_embedding,pointset=pointset,centerset=centerset)
    return enc

def get_ffn(input_dim,f_act,context_str=''):
    if config.use_layn == "T":
        use_layn = True
    else:
        use_layn = False
    if config.skip_connection == "T":
        skip_connection = True
    else:
        skip_connection = False
    return MultiLayerFeedForwardNN(
        input_dim=input_dim,
        output_dim=config.spa_embed_dim,
        num_hidden_layers=config.num_hidden_layer,
        dropout_rate=config.dropout,
        hidden_dim=config.hidden_dim,
        activation=f_act,
        use_layernormalize=use_layn,
        skip_connection=skip_connection,
        context_str=context_str
    )

def get_spa_encoder(spa_enc_type,spa_embed_dim,coord_dim=2,frequency_num=16,max_radius=10000,min_radius=1,f_act='sigmoid',freq_init='geometric',use_postmat='T'):
    if spa_enc_type == 'gridcell':
        ffn = get_ffn(input_dim=int(4*frequency_num),f_act=f_act,context_str='Grid')
        spa_enc = RelationEncoder(spa_embed_dim,coord_dim=coord_dim,frequency_num=frequency_num,max_radius=max_radius,min_radius=min_radius,freq_init=freq_init,ffn=ffn)
    elif spa_enc_type == 'theory':
        ffn = get_ffn(input_dim=int(4*frequency_num),f_act=f_act,context_str='Theory')
        spa_enc = TheoryRelationEncoder(spa_embed_dim,coord_dim=coord_dim,frequency_num=frequency_num,max_radius=max_radius,min_radius=min_radius,freq_init=freq_init,ffn=ffn)
    else:
        raise Exception("Space encoder function no support!")
    return spa_enc

# def get_context_decoder(query_dim,key_dim,spa_embed_dim,g_spa_embed_dim,have_query_embed=True,num_attn=1,activation='relu',f_activation='sigmoid',layn='T',use_postmat='T',dropout=0.5):
#     if layn=='T':
#         layernorm = True
#     else:
#         layernorm = False
#     if use_postmat == 'T':
#         use_post_mat = True
#     else:
#         use_post_mat = False
#     dec = Decoder(query_dim=query_dim,key_dim=key_dim,spa_embed_dim=spa_embed_dim,have_query_embed=have_query_embed,num_attn=num_attn,
#                   activation=activation,f_activation=f_activation,layernorm=layernorm,use_post_mat=use_post_mat,dropout=dropout)
#     return dec

# def get_enc_dec(model_type, pointset,centerset, enc, spa_enc = None, 
#                 g_spa_enc = None, g_spa_dec = None, init_dec=None, dec=None, joint_dec=None, 
#                 activation = "sigmoid", num_context_sample = 10, num_neg_resample = 10):
#     enc_dec = EncoderDecoder(pointset=pointset,
#                             centerset=centerset,
#                             enc=enc,
#                             spa_enc=spa_enc,
#                             init_dec=init_dec,
#                             dec=dec,
#                             activation=activation,
#                             num_poi_sample=num_context_sample,
#                             num_neg_resample=num_neg_resample)
#     return enc_dec