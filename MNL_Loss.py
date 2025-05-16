import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import itertools

EPS = 1e-2
esp = 1e-12

class Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))

        return torch.mean(loss)

class Fidelity_Loss_distortion(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss_distortion, self).__init__()

    def forward(self, p, g):
        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i]
            g_i = g[:, i]
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = torch.sqrt(p_i * g_i + esp)
            loss = loss + loss_i
        loss = 1 - loss
        #loss = loss / p.size(1)
        return torch.mean(loss)


class Multi_Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Multi_Fidelity_Loss, self).__init__()

    def forward(self, p, g):

        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i]
            g_i = g[:, i]
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = 1 - (torch.sqrt(p_i * g_i + esp) + torch.sqrt((1 - p_i) * (1 - g_i) + esp))
            loss = loss + loss_i
        loss = loss / p.size(1)

        return torch.mean(loss)

class Multi_Fidelity_Loss2(torch.nn.Module):

    def __init__(self):
        super(Multi_Fidelity_Loss2, self).__init__()

    def forward(self, p, g):
        total_loss = 0
        for i in range(g.size(0)):
            non_zero_terms = torch.nonzero(g[i, :])
            loss = 0
            for j in range(len(non_zero_terms)):
                p_ij = p[i, non_zero_terms[j]]
                g_ij = g[i, non_zero_terms[j]]

                g_ij = g_ij.view(-1, 1)
                p_ij = p_ij.view(-1, 1)
                loss_ij = 1 - (torch.sqrt(p_ij * g_ij + esp) + torch.sqrt((1 - p_ij) * (1 - g_ij) + esp))

                loss = loss + loss_ij
            if len(non_zero_terms) != 0:
                loss = loss / len(non_zero_terms)
            total_loss = total_loss + loss

        total_loss = total_loss / g.size(0)
        return total_loss

eps = 1e-12


def loss_m(y_pred, y):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    preds = y_pred-(y_pred + 10).t()
    gts = y.t() - y
    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    return torch.sum(F.relu(preds * torch.sign(gts))) / preds.size(0)
    #return torch.sum(F.relu((y_pred-(y_pred + 10).t()) * torch.sign((y.t()-y)))) / y_pred.size(0) / (y_pred.size(0)-1)


def loss_m2(y_pred, y, gstd):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    preds = y_pred-y_pred.t()
    gts = y - y.t()
    g_var = gstd * gstd + gstd.t() * gstd.t() + eps

    #signed = torch.sign(gts)

    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    g_var = g_var[triu_indices[0], triu_indices[1]]
    #signed = signed[triu_indices[0], triu_indices[1]]

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    g = 0.5 * (1 + torch.erf(gts / torch.sqrt(g_var)))
    p = 0.5 * (1 + torch.erf(preds / constant))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss


def loss_m3(y_pred, y):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    y_pred = y_pred.unsqueeze(1)
    y = y.unsqueeze(1)
    preds = y_pred-y_pred.t()
    gts = y - y.t()

    #signed = torch.sign(gts)

    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    g = 0.5 * (torch.sign(gts) + 1)

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    p = 0.5 * (1 + torch.erf(preds / constant))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss

def loss_m4(y_pred_all, per_num, y_all):
    """prediction monotonicity related loss"""
    loss = 0
    pos_idx = 0
    for task_num in per_num:
        y_pred = y_pred_all[pos_idx:pos_idx+task_num]
        y = y_all[pos_idx:pos_idx+task_num]
        pos_idx = pos_idx + task_num

        #assert y_pred.size(0) > 1  #
        if y_pred.size(0) == 0:
            continue
        y_pred = y_pred.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()
        gts = y - y.t()

        # signed = torch.sign(gts)

        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        loss += torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    loss = loss / len(per_num)

    return loss


def loss_hybrid_focal2(y_pred_all, per_num1, y_all, pred, per_num2, gt, gamma=0):
    """prediction monotonicity related loss"""
    #loss = 0
    pos_idx = 0
    loss = 0
    real_bs = 0
    for task_num in per_num1:
        y_pred = y_pred_all[pos_idx:pos_idx+task_num]
        y = y_all[pos_idx:pos_idx+task_num]
        pos_idx = pos_idx + task_num

        if y_pred.size(0) == 0:
            continue
        if len(y_pred.size()) == 1:
            y_pred = y_pred.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()
        gts = y - y.t()

        # signed = torch.sign(gts)

        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        #loss_t = (1 - ((1-p) ** gamma) * (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp)))

        loss_t = ((1-p)**gamma * (1 - torch.sqrt(p*g + esp)) * g
                  + (p)**gamma * (1 - torch.sqrt((1 - p) * (1 - g) + esp)) * (1 - g))

        real_bs = real_bs + loss_t.size(0)
        loss_t = torch.sum(loss_t, dim=0)
        loss = loss + loss_t

    pos_idx = 0
    for task_num in per_num2:
        y_pred = pred[pos_idx:pos_idx+task_num]
        y = gt[pos_idx:pos_idx+task_num]
        pos_idx = pos_idx + task_num

        y_pred = y_pred.view(-1, 1)
        y = y.view(-1, 1)

        #loss_t = (1 - ((1 - y_pred) ** gamma)*(torch.sqrt(y_pred * y + esp) + torch.sqrt((1 - y_pred) * (1 - y) + esp)))
        loss_t = ((1-y_pred)**gamma * (1 - torch.sqrt(y_pred*y + esp)) * y
                  + (y_pred)**gamma * (1 - torch.sqrt((1 - y_pred) * (1 - y) + esp)) * (1 - y))
        real_bs = real_bs + loss_t.size(0)
        loss_t = torch.sum(loss_t, dim=0)
        loss = loss + loss_t
    #print('sum_loss:{} real_bs:{}'.format(loss.item(), real_bs))
    loss = loss / real_bs
    return loss


def loss_focal(y_pred_all, per_num, y_all, gamma=1):
    """prediction monotonicity related loss"""
    #loss = 0
    pos_idx = 0
    loss = 0
    real_bs = 0
    for task_num in per_num:
        y_pred = y_pred_all[pos_idx:pos_idx+task_num]
        y = y_all[pos_idx:pos_idx+task_num]
        pos_idx = pos_idx + task_num

        if y_pred.size(0) == 0:
            continue
        if len(y_pred.size()) == 1:
            y_pred = y_pred.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()
        gts = y - y.t()

        # signed = torch.sign(gts)

        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        #loss_t = (1 - ((1-p) ** gamma) * (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp)))

        loss_t = ((1-p)**gamma * (1 - torch.sqrt(p*g + esp)) * g
                  + (p)**gamma * (1 - torch.sqrt((1 - p) * (1 - g) + esp)) * (1 - g))

        real_bs = real_bs + loss_t.size(0)
        loss_t = torch.sum(loss_t, dim=0)
        loss = loss + loss_t

    #print('sum_loss:{} real_bs:{}'.format(loss.item(), real_bs))
    loss = loss / real_bs
    return loss


def plcc_loss(y, y_pred):
    y = y.to(y_pred.dtype)
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    # print(y.shape)
    # print(y_pred.shape)
    loss0 = F.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = F.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2)


def loss_mix(y_pred_all, per_num, y_all):
    """prediction monotonicity related loss"""
    loss = 0
    pos_idx = 0
    for task_num in per_num:
        y_pred = y_pred_all[pos_idx:pos_idx+task_num]
        y = y_all[pos_idx:pos_idx+task_num]

        plcc = plcc_loss(y, y_pred)

        pos_idx = pos_idx + task_num

        #assert y_pred.size(0) > 1  #
        if y_pred.size(0) == 0:
            continue
        y_pred = y_pred.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()
        gts = y - y.t()

        # signed = torch.sign(gts)

        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        loss += 0.5*(torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp)))) + plcc)

    loss = loss / len(per_num)

    return loss