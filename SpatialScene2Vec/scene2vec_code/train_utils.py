import random
import numpy as np
from numpy.core.numeric import full
from numpy.ma.core import maximum_fill_value
import torch

def check_conv(vals,window=3,tol=1e-6):
    if len(vals)<2*window:
        return False
    conv = np.mean(vals[-window:]) - np.mean(vals[-2*window:-window])
    return conv<tol

def update_loss(loss,losses,ema_loss,ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1-ema_alpha)*ema_loss+ema_alpha*loss
    return losses,ema_loss

def run_eval(model,ng_list,iteration,logger,batch_size=1000,do_full_eval=True):
    if model is not None:
        model.eval()

    full_ranks = []
    offset = 0
    while offset<len(ng_list):
        max_index = min(offset+batch_size,len(ng_list))
        batch_ng_list = ng_list[offset:max_index]
        ranks = get_batch_ranks(model,batch_ng_list,do_full_eval)
        full_ranks += ranks
        offset+=batch_size
    mrr,hit1,hit5,hit10 = eval_stat(full_ranks)
    return mrr,hit1,hit5,hit10

def run_eval_per_type(model,pointset,centerset,ng_list,typeid2root=None,batch_size=1000,do_full_eval=True):
    if model is not None:
        model.eval()
    full_ranks = []
    offset = 0
    while offset < len(ng_list):
        max_index = min(offset+batch_size,len(ng_list))
        batch_ng_list = ng_list[offset:max_index]
        ranks = get_batch_ranks(model,batch_ng_list,do_full_eval)
        full_ranks += ranks
        offset+=batch_size
    type2rank = dict()
    for i,ng in enumerate(ng_list):
        type_list = list(ng.center_pt)
        if typeid2root is not None:
            type_list = list(set([typeid2root[typeid] for typeid in type_list]))
        for pt_type in type_list:
            if pt_type not in type2rank:
                type2rank[pt_type] = []
            type2rank[pt_type].append(full_ranks[i])
    type2mrr = dict()
    type2hit1 = dict()
    type2hit5 = dict()
    type2hit10 = dict()
    for pt_type in type2rank:
        type2mrr[pt_type],type2hit1[pt_type],type2hit5[pt_type],type2hit10[pt_type] = eval_stat(type2rank[pt_type])

    return type2mrr,type2hit1,type2hit5,type2hit10

def eval_stat(full_ranks):
    num_sample = len(full_ranks)

    mrr = 0.0
    hit1 = 0.0
    hit5 = 0.0
    hit10 = 0.0

    for rank in full_ranks:
        mrr += 1.0/rank
        if rank <= 1:
            hit1 += 1.0
        if rank <= 5:
            hit5 += 1.0
        if rank <= 10:
            hit10 += 1.0
    
    mrr /= num_sample
    hit1 /= num_sample
    hit5 /= num_sample
    hit10 /= num_sample
    return mrr,hit1,hit5,hit10

def get_batch_ranks(model,ng_list,do_full_eval=True):
    if model:
        pos,neg = model.get_batch_scores(ng_list,do_full_eval)
        scores = torch.cat((pos.unsqueeze(1),neg),dim=1)
        scores = np.array(scores.data.tolist())
    else:
        batch_size = len(ng_list)
        if do_full_eval:
            num_neg_sample = 100
        else:
            num_neg_sample = 10
        pos = np.random.randn(batch_size,1)
        neg = np.random.randn(batch_size,num_neg_sample)
        scores = np.concatenate((pos,neg),axis=1)

    batch_size,num_pt = scores.shape
    ranks = num_pt - np.argmin(np.argsort(scores,axis=-1),axis=-1)
    return list(ranks)

def run_batch(train_ng_list,enc_dec,iter_count,batch_size,do_full_eval):
    n = len(train_ng_list)
    start = (iter_count*batch_size)%n
    end = min(((iter_count+1)*batch_size)%n,n)
    end = n if end <= start else end
    ng_list = train_ng_list[start:end]
    loss = enc_dec.softmax_loss(ng_list,do_full_eval)
    return loss

def run_train(model,optimizer,train_ng_list,val_ng_list,test_ng_list,logger,
              max_iter=int(10e7),batch_size=512,log_every=100,val_every=1000,tol=1e-6,model_file=None):
    ema_loss = None
    vals = []
    losses = []

    ema_loss_val = None
    losses_val = []

    if model is not None:
        random.shuffle(train_ng_list)
        for i in range(max_iter):
            model.train()
            optimizer.zero_grad()
            loss = run_batch(train_ng_list,model,i,batch_size,do_full_eval=True)
            losses,ema_loss = update_loss(loss.item(),losses,ema_loss)
            loss.backward()
            optimizer.step()

            loss_val = run_batch(val_ng_list,model,i,batch_size,do_full_eval=False)
            losses_val,ema_loss_val = update_loss(loss_val.item(),losses_val,ema_loss_val)

            if i%log_every == 0:
                logger.info("Iter {:d}; Train ema_loss {:f}".format(i,ema_loss))
                logger.info("Iter {:d}; Validate ema_loss {:f}".format(i,ema_loss_val))
            
            if i>=val_every and i%val_every==0:
                mrr, hit1, hit5, hit10 = run_eval(model, random.sample(train_ng_list, len(val_ng_list)), i, logger, do_full_eval = True)
                logger.info("Iter: {:d}; 10 Neg, Train MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))


                mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = False)
                logger.info("Iter: {:d}; 10 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

                mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = True)
                logger.info("Iter: {:d}; 100 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))
                
                vals.append(mrr)
                if not model_file is None:
                    torch.save(model.state_dict(), model_file)
    else:
        i = 0
    mrr, hit1, hit5, hit10 = run_eval(model, random.sample(train_ng_list, len(val_ng_list)), i, logger, do_full_eval = True)
    logger.info("Iter: {:d}; 10 Neg, Train MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

    mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = False)
    logger.info("Iter: {:d}; 10 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))

    mrr, hit1, hit5, hit10 = run_eval(model, val_ng_list, i, logger, do_full_eval = True)
    logger.info("Iter: {:d}; 100 Neg, Validate MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr, hit1, hit5, hit10))
    
    mrr_, hit1_, hit5_, hit10_ = run_eval(model, test_ng_list, i, logger, do_full_eval = False)
    logger.info("Iter: {:d}; 10 Neg, Test MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr_, hit1_, hit5_, hit10_))

    mrr_, hit1_, hit5_, hit10_ = run_eval(model, test_ng_list, i, logger, do_full_eval = True)
    logger.info("Iter: {:d}; 100 Neg, Test MRR: {:f}, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(i, mrr_, hit1_, hit5_, hit10_))

    