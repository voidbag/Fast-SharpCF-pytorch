import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from collections import defaultdict
#from tensorboardX import SummaryWriter

import config
from SharpCF import SharpCF
import shutil
import pandas as pd
from copy import deepcopy
from logger import TimeLogger

from BPRpytorch import evaluate
import slow_data_utils



parser = argparse.ArgumentParser()
parser.add_argument("--lr",
        type=float,
        default=0.01,
        help="learning rate")
parser.add_argument("--lamda",
        type=float,
        default=0.001,
        help="model regularization rate")
parser.add_argument("--batch_size",
        type=int,
        default=4096,
        help="batch size for training")
parser.add_argument("--epochs",
        type=int,
        default=50,
        help="training epoches")
parser.add_argument("--top_k",
        type=int,
        default=10,
        help="compute metrics@top_k")
parser.add_argument("--factor_num",
        type=int,
        default=32,
        help="predictive factors numbers in the model")
parser.add_argument("--num_ng",
        type=int,
        default=4,
        help="sample negative items for training")
parser.add_argument("--test_num_ng",
        type=int,
        default=99,
        help="sample part of negative items for testing")
parser.add_argument("--out",
        default=True,
        help="save model or not")
parser.add_argument("--gpu",
        type=str,
        default="0",
        help="gpu card ID")

parser.add_argument("--free_warm",
        type=int,
        default=200)

parser.add_argument("--free_szwin",
        type=int,
        default=3)

parser.add_argument("--free_lambda",
        type=float,
        default=0.1)

parser.add_argument("--chk_period",
        type=int,
        default=100)

parser.add_argument("--out_dir",
        type=str,
        default="./out_report_post")

args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat = slow_data_utils.load_all()

# construct the train and test datasets
    
train_dataset = slow_data_utils.SlowBPRData(
                train_data, item_num, train_mat, args.num_ng, is_training=True, reuse_order=True)
test_dataset = slow_data_utils.SlowBPRData(
                test_data, item_num, train_mat, 0, is_training=False, reuse_order=False)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
train_loader.dataset.ng_sample()
order = torch.tensor(train_loader.dataset.order).cuda() 
model = SharpCF(user_num, item_num, args.factor_num, (args.free_szwin, len(train_dataset)), order)
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lamda)
# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################



num_batch = (len(train_dataset) + (args.batch_size - 1)) // args.batch_size
print(f"num_batch: {num_batch}")


free_warm = args.free_warm
free_szwin = args.free_szwin
free_lambda = args.free_lambda
t_free_lambda = torch.tensor((free_lambda), dtype=torch.float32, requires_grad=False).cuda()
t_batch_size = torch.tensor((args.batch_size), dtype=torch.int32, requires_grad=False).cuda()

best_hr = 0
best_ndcg = 0
best_hr_epoch = 0
best_ndcg_epoch = 0

d = "batch_{}_lr_{}_epochs_{}_freewarm_{}_freelambda_{}".format(args.batch_size, args.lr, args.epochs, args.free_warm, args.free_lambda)
dir_home = os.path.join(args.out_dir, d)

os.makedirs(dir_home, exist_ok=True)

li_loss = list()
li_ndcg = list()
li_hr = list()
li_epoch = list()
best_state = None
li_dict = list()

for epoch in range(args.epochs):
    start_time = time.time()
    logger = TimeLogger()
    logger.reset()
    model.train()
    logger.log("01.train()")

    train_dataset.ng_sample()
    logger.log("02.neg_sample")
    
    logger.reset()
    loss_tr = 0
    idx = 0
    for user, item_i, item_j in train_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()
        start = idx * args.batch_size
        end = min(start + args.batch_size, len(train_dataset)) 
        logger.log("03.get_batch")
        model.zero_grad()
        logger.log("04.zero_grad")

        prediction_i, prediction_j = model(user, item_i, item_j)
        loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        loss_prev = loss

        idx_score_epoch = (epoch - (free_szwin - 1)) % free_szwin
        if epoch >= free_warm:
            diff = prediction_i - model.window[idx_score_epoch, start:end]
            loss = loss + (t_free_lambda / (end - start + 1)) * torch.dot(diff, diff)
        model.window[idx_score_epoch, start:end] = prediction_i.detach()

        logger.log("05.forward")
        loss.backward()
        logger.log("05.backward")

        optimizer.step()
        loss_tr += (loss.item() / (len(train_dataset)))
        logger.log("06.step")
        idx += 1
        # writer.add_scalar('data/loss', loss.item(), count)
        #count += 1
    logger.reset()
    model.eval()
    logger.log("07.eval()")
    HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
    hr_mean, ndcg_mean = np.mean(HR), np.mean(NDCG)
    stat_loss = loss_tr
    logger.log("08.hr_ndcg")
    
    li_loss.append(stat_loss)
    li_ndcg.append(ndcg_mean)
    li_hr.append(hr_mean)
    li_epoch.append(epoch)
    li_dict.append(logger.export_dict())

    if (epoch + 1) % args.chk_period == 0 or (epoch + 1) == free_warm or (args.epochs - 1) == epoch:
        path_model = os.path.join(dir_home, f"{epoch}.pth")
        #model.update_order(train_dataset.order)
        torch.save(model.state_dict(), path_model)
        dict_ts = defaultdict(list)
        li_keys = sorted(li_dict[0].keys())
        for d in li_dict:
            for k in li_keys:
                dict_ts[k].append(d[k])

        df_stat = pd.DataFrame(dict(epoch=li_epoch, loss=li_loss, ndcg=li_ndcg, hr=li_hr))
        df_stat = pd.concat([df_stat, pd.DataFrame(dict_ts)], axis=1)
        path_stat = os.path.join(dir_home, "stat.pkl")
        if os.path.exists(path_stat):
            shutil.move(path_stat, path_stat + ".old")
        df_stat.to_pickle(path_stat)


    elapsed_time = time.time() - start_time
    print("Epoch {:03d}".format(epoch) + " time : " +
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)) + "." + str(elapsed_time).split(".")[1][:2])
    print("--HR: {:.3f}\tNDCG: {:.3f}, LOSS: {:.3f}".format(hr_mean, ndcg_mean, stat_loss))
    logger.print_log()
    #
    if hr_mean > best_hr:
        best_hr, best_hr_epoch = hr_mean, epoch
        best_hr_state = deepcopy(model.state_dict())

    if ndcg_mean > best_ndcg:
        best_ndcg, best_ndcg_epoch = ndcg_mean, epoch
        best_ndcg_state = deepcopy(model.state_dict())

path_model = os.path.join(dir_home, f"best_hr_epoch_{best_hr_epoch}.pth")
torch.save(best_hr_state, path_model)

path_model = os.path.join(dir_home, f"best_ndcg_epoch_{best_ndcg_epoch}.pth")
torch.save(best_ndcg_state, path_model)



print("End. Best epoch {:03d}: HR = {:.3f}".format(best_hr_epoch, best_hr))
print("End. Best epoch {:03d}: NDCG = {:.3f}".format(best_ndcg_epoch, best_ndcg))
