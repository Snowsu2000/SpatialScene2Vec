from scene2vec_code.utils import *
from scene2vec_code.train_utils import *
# from utils import *
from torch import optim
import time
import os
import scene2vec_code.config as config
from scene2vec_code.model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = ('cpu')
class Trainer():
    def __init__(self,poiset,centerset,ng_pair_list,feature_embedding,console=True) -> None:
        self.poiset = poiset
        self.centerset = centerset
        self.ng_pair_list = np.array(ng_pair_list.copy())
        self.feature_embedding = feature_embedding
        self.log_name = 'Model_' + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
        self.log_file = os.path.abspath('.') + '/log/' + self.log_name + '.log'
        self.model_file = os.path.abspath('.')+'/model_save/' + self.log_name + '.pth'
        self.logger = setup_logging(self.log_file,console=console,filemode='a')
        self.feature_encoder_q = get_feature_encoder(feature_embed_lookup=poiset.feature_embed_lookup,feature_embedding=feature_embedding,pointset=poiset,centerset=centerset)
        self.feature_encoder_k = get_feature_encoder(feature_embed_lookup=poiset.feature_embed_lookup,feature_embedding=feature_embedding,pointset=poiset,centerset=centerset)
        self.model_type = ['gridcell','theory']
        self.position_encoder_q = get_spa_encoder(spa_enc_type=self.model_type[0],
                                                spa_embed_dim=config.spa_embed_dim,
                                                coord_dim=2,
                                                frequency_num=config.freq,
                                                max_radius=config.max_radius,
                                                min_radius=config.min_radius,
                                                f_act=config.spa_f_act[1],
                                                freq_init=config.freq_init)
        self.position_encoder_k = get_spa_encoder(spa_enc_type=self.model_type[0],
                                                spa_embed_dim=config.spa_embed_dim,
                                                coord_dim=2,
                                                frequency_num=config.freq,
                                                max_radius=config.max_radius,
                                                min_radius=config.min_radius,
                                                f_act=config.spa_f_act[1],
                                                freq_init=config.freq_init)
        self.encoder_q = Scene2Vec(poiset=self.poiset,
                                 centerset=self.centerset,
                                 position_encoder=self.position_encoder_q,
                                 feature_encoder=self.feature_encoder_q)
        self.encoder_k = Scene2Vec(poiset=self.poiset,
                                 centerset=self.centerset,
                                 position_encoder=self.position_encoder_k,
                                 feature_encoder=self.feature_encoder_k)
        self.cl_model = ConstraintModel(encoder_q=self.encoder_q,encoder_k=self.encoder_k)
        self.cl_model.to(device=device)
        
        
        opt = config.opt[0]
        if opt=="sgd":
            self.self.optimizer = optim.SGD(self.cl_model.parameters(),lr=config.lr,momentum=0)
        elif opt=="adam":
            self.optimizer = optim.Adam(self.cl_model.parameters(),lr=config.lr)

        print("Create model from {}".format(self.log_name+'.pth'))
        self.logger.info("Save file at {}".format(self.log_name+".pth"))

    def run_train(self,max_iter=int(10e7),batch_size=512,log_every=100,tol=1e-6):
        ema_loss = None
        vals = []
        losses = []

        ema_loss_val = None
        losses_val = []
        
        if self.cl_model is not None:
            random.shuffle(self.ng_pair_list)
            for i in range(max_iter):
                self.cl_model.train()
                self.optimizer.zero_grad()
                
                loss,acc = self.run_batch(i,batch_size,do_full_eval=True)
                losses,ema_loss = update_loss(loss.item(),losses,ema_loss)
                loss.backward()
                self.optimizer.step()

                # loss_val = run_batch(val_ng_list,self.cl_model,i,batch_size,do_full_eval=False)
                # losses_val,ema_loss_val = update_loss(loss_val.item(),losses_val,ema_loss_val)
                # print(loss,acc)
                if i%log_every == 0:
                    self.logger.info("Iter {:d}; Train ema_loss {:f}".format(i,ema_loss))
                    self.logger.info("Iter: {:d}; Train HIT@1: {:f}, HIT@3: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, acc[0][0],acc[1][0],acc[2][0],acc[3][0]))
                    # break
                    if not self.model_file is None:
                        torch.save(self.cl_model.state_dict(), self.model_file)
        else:
            i = 0
       
    
    def run_batch(self,iter_count,batch_size,do_full_eval):
        n = len(self.ng_pair_list)
        start = (iter_count*batch_size)%n
        end = min(((iter_count+1)*batch_size)%n,n)
        end = n if end <= start else end
        ng_list_q,ng_list_k = self.ng_pair_list[start:end,0],self.ng_pair_list[start:end,0]
        with torch.cuda.amp.autocast(True):
            loss = self.cl_model(ng_list_q,ng_list_k)
        # print(loss,loss.requires_grad)
        return loss#,acc

    def train(self):
        self.run_train(
            max_iter=config.max_iter,
            batch_size=config.batch_size,
            log_every=config.log_every,
            tol=config.tol
        )

