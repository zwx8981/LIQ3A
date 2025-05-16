import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import random
import time
from MNL_Loss import loss_m4, Multi_Fidelity_Loss, Fidelity_Loss_distortion, loss_mix
import scipy.stats
from utils import _preprocess_siglip_train, _preprocess_siglip, set_dataset_qonly, set_dataset_llie_naflex
import os
import pickle
from modular import ViTSigLIP, ViTSigLIP_naflex
from torch.amp import GradScaler
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def emd_loss(pred, target, r=2):
    """
    Args:
        pred (Tensor): of shape (N, C). Predicted tensor.
        target (Tensor): of shape (N, C). Ground truth tensor.
        r (float): norm level, default l2 norm.
    """
    loss = torch.abs(torch.cumsum(pred, dim=-1) - torch.cumsum(target, dim=-1))**r
    loss = loss**(1. / r)
    #loss = loss.mean(dim=-1)**(1. / r)
    return loss


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def compute_metrics(y_pred, y):
    '''
    compute metrics btw predictions & labels
    '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute  PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return SRCC, KRCC, PLCC, RMSE

##############################general setup####################################
llie_set = '../IQA_Database/LLIE_dataset'
llie_set2 = '../IQA_Database/SDSD-enhance/imgs'

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# initial_lr = 5e-6
# num_epoch = 30
bs = 16

initial_lr = 5e-6
num_epoch = 8

train_patch = 3

loss_img2 = Fidelity_Loss_distortion()
loss_scene = Multi_Fidelity_Loss()

##############################general setup####################################

preprocess2 = _preprocess_siglip()
preprocess3 = _preprocess_siglip_train()

def train(model, best_result, best_epoch, srcc_dict, plcc_dict, emd_dict):
    start_time = time.time()
    beta = 0.9
    running_loss = 0 if epoch == 0 else train_loss[-1]
    running_duration = 0.0
    num_steps_per_epoch = 200
    local_counter = epoch * num_steps_per_epoch + 1
    model.train()
    loaders = []
    for loader in train_loaders:
        loaders.append(iter(loader))

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    for step in range(num_steps_per_epoch):
        #total_loss = 0
        all_batch = []
        spatial_batch = []
        gmos_batch = []
        num_sample_per_task = []
        gt_dists = []
        for dataset_idx, loader in enumerate(loaders, 0):
            try:
                sample_batched = next(loader)
            except StopIteration:
                loader = iter(train_loaders[dataset_idx])
                sample_batched = next(loader)
                loaders[dataset_idx] = loader

            x, gmos, dists = sample_batched


            #x = x.to(device)
            gmos = gmos.to(device)
            gmos_batch.append(gmos)
            num_sample_per_task.append(gmos.size(0))
            gt_dists.append(dists.to(device))

            # preserve all samples into a batch, will be used for optimization of scene and distortion type later
            all_batch = all_batch + x

        gmos_batch = torch.cat(gmos_batch, dim=0)
        gt_dists = torch.cat(gt_dists, dim=0)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits_quality, logits_distortion = model(all_batch)

            total_loss = loss_mix(logits_quality, num_sample_per_task,
                                       gmos_batch.detach()).mean() + loss_img2(
                logits_distortion[:gt_dists.size(0), :], gt_dists.detach()).mean()

            total_loss = loss_mix(logits_quality, num_sample_per_task, gmos_batch.detach()).mean()


            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # statistics
        running_loss = beta * running_loss + (1 - beta) * total_loss.data.item()
        loss_corrected = running_loss / (1 - beta ** local_counter)

        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** local_counter)
        examples_per_sec = len(all_batch) / duration_corrected
        format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (epoch, step + 1, num_steps_per_epoch, loss_corrected,
                            examples_per_sec, duration_corrected))

        local_counter += 1
        start_time = time.time()

        train_loss.append(loss_corrected)

    all_result = {'val':{}, 'test':{}}
    if (epoch >= 0):

        srcc1, plcc1, emd1, = eval(llie_val_loader, phase='val', dataset='llie')
        srcc11, plcc11, emd11 = eval(llie_test_loader, phase='test', dataset='llie')


        srcc5, plcc5, emd5 = eval(llie_val_loader2, phase='val', dataset='llie2')
        srcc55, plcc55, emd55 = eval(llie_test_loader2, phase='test', dataset='llie2')

        srcc_avg = (srcc1 + srcc5) / 2

        #srcc_avg = srcc5

        current_avg = srcc_avg

        if current_avg > best_result['avg']:
            print('**********New overall best!**********')
            best_epoch['avg'] = epoch
            best_result['avg'] = current_avg
            srcc_dict['llie'] = srcc11
            srcc_dict['llie2'] = srcc55


            plcc_dict['llie'] = plcc11
            plcc_dict['llie2'] = plcc55

            emd_dict['llie'] = emd11
            emd_dict['llie2'] = emd55

            ckpt_name = os.path.join('checkpoints', str(session+1), 'siglip_joint2.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'all_results':all_result
            }, ckpt_name)  # just change to your preferred folder/filename

    return best_result, best_epoch, srcc_dict, plcc_dict, emd_dict


def eval(loader, phase, dataset):
    model.eval()
    q_mos = []
    q_hat = []
    emds = []
    for step, sample_batched in enumerate(loader, 0):

        x, gmos, gdists = sample_batched
        gmos = gmos.to(device)
        gdists = gdists.to(device)
        q_mos = q_mos + gmos.cpu().tolist()
        # Calculate features
        with torch.no_grad():
            logits_quality, logits_distortion = model(x)

        q_hat = q_hat + logits_quality.cpu().tolist()

        emd = emd_loss(logits_distortion, gdists)
        emds = emds + emd.cpu().tolist()

    srcc, _, plcc, _ = compute_metrics(q_hat, q_mos)
    emds = np.array(emds)
    emds = np.mean(emds)

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)
    model.train()
    return srcc, plcc, emds

num_workers = 8
for session in range(0,1):
    model = ViTSigLIP_naflex()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)


    train_loss = []
    start_epoch = 0

    best_result = {'avg': 0.0}
    best_epoch = {'avg': 0}
    # avg
    srcc_dict = {'llie': 0.0, 'llie2': 0.0}
    plcc_dict = {'llie': 0.0, 'llie2': 0.0}
    emd_dict = {'llie': 0.0, 'llie2': 0.0}


    llie_train_csv = os.path.join('./llie_train.csv')
    llie_val_csv = os.path.join('./llie_val.csv')
    llie_test_csv = os.path.join('./llie_test.csv')

    llie_train_loader = set_dataset_llie_naflex(llie_train_csv, 16, llie_set,  num_workers, preprocess3,
                                              train_patch, False, set=0)
    llie_val_loader = set_dataset_llie_naflex(llie_val_csv, 16, llie_set, num_workers, preprocess2,
                                            15, True, set=1)
    llie_test_loader = set_dataset_llie_naflex(llie_test_csv, 16, llie_set, num_workers, preprocess2,
                                             15, True, set=2)

    llie_train_csv2 = os.path.join('./llie_train2.csv')
    llie_val_csv2 = os.path.join('./llie_val2.csv')
    llie_test_csv2 = os.path.join('./llie_test2.csv')

    llie_train_loader2 = set_dataset_llie_naflex(llie_train_csv2, 16, llie_set2 , num_workers, preprocess3,
                                              train_patch, False, set=0)
    llie_val_loader2 = set_dataset_llie_naflex(llie_val_csv2, 16, llie_set2, num_workers, preprocess2,
                                            15, True, set=1)
    llie_test_loader2 = set_dataset_llie_naflex(llie_test_csv2, 16, llie_set2, num_workers, preprocess2,
                                             15, True, set=2)

    
    
    train_loaders = [llie_train_loader, llie_train_loader2]
    result_pkl = {}
    scaler = GradScaler()
    for epoch in range(0, num_epoch):
        best_result, best_epoch, srcc_dict,  plcc_dict, emd_dict = train(model, best_result, best_epoch, srcc_dict, plcc_dict, emd_dict)
        scheduler.step()

        print('...............current average best...............')
        print('best average epoch:{}'.format(best_epoch['avg']))
        print('best average result:{}'.format(best_result['avg']))
        for dataset in srcc_dict.keys():
            print_text = dataset + ':' + 'srcc:{} '.format(srcc_dict[dataset]) + 'plcc:{} '.format(
                plcc_dict[dataset]) + 'emd:{}'.format(emd_dict[dataset])
            print(print_text)


