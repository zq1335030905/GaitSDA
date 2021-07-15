import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.float().cuda()
    else:
        return variable

def kl_loss(code):
    return torch.mean(torch.pow(code, 2))


def pairwise_cosine_similarity(seqs_i, seqs_j):
    # seqs_i, seqs_j: [batch, statics, channel]
    n_statics = seqs_i.size(1)
    seqs_i_exp = seqs_i.unsqueeze(2).repeat(1, 1, n_statics, 1)
    seqs_j_exp = seqs_j.unsqueeze(1).repeat(1, n_statics, 1, 1)
    return F.cosine_similarity(seqs_i_exp, seqs_j_exp, dim=3)


def temporal_pairwise_cosine_similarity(seqs_i, seqs_j):
    # seqs_i, seqs_j: [batch, channel, time]
    seq_len = seqs_i.size(2)
    seqs_i_exp = seqs_i.unsqueeze(3).repeat(1, 1, 1, seq_len)
    seqs_j_exp = seqs_j.unsqueeze(2).repeat(1, 1, seq_len, 1)
    return F.cosine_similarity(seqs_i_exp, seqs_j_exp, dim=1)


def consecutive_cosine_similarity(seqs):
    # seqs: [batch, channel, time]
    seqs_roll = torch.roll(seqs, shifts = -1, dims = 2)[:,:,1:]
    seqs = seqs[:,:,:-1]
    return torch.mean(abs((torch.cosine_similarity(seqs, seqs_roll, dim=2))))


def triplet_margin_loss(seqs_a, seqs_b, neg_range=(0.0, 0.5), margin=0.2):
    # seqs_a, seqs_b: [batch, channel, time]

    neg_start, neg_end = neg_range
    batch_size, _, seq_len = seqs_a.size()
    n_neg_all = seq_len ** 2
    n_neg = int(round(neg_end * n_neg_all))
    n_neg_discard = int(round(neg_start * n_neg_all))

    batch_size, _, seq_len = seqs_a.size()
    sim_aa = temporal_pairwise_cosine_similarity(seqs_a, seqs_a)
    sim_bb = temporal_pairwise_cosine_similarity(seqs_b, seqs_a)
    sim_ab = temporal_pairwise_cosine_similarity(seqs_a, seqs_b)
    sim_ba = sim_ab.transpose(1, 2)

    diff_ab = (sim_ab - sim_aa).reshape(batch_size, -1)
    diff_ba = (sim_ba - sim_bb).reshape(batch_size, -1)
    diff = torch.cat([diff_ab, diff_ba], dim=0)
    diff, _ = diff.topk(n_neg, dim=-1, sorted=True)
    diff = diff[:, n_neg_discard:]

    loss = diff + margin
    loss = loss.clamp(min=0.)
    loss = loss.mean()

    return loss

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),device=targets.device).fill_(smoothing /(n_classes-1)).scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        # print(inputs, targets)
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        lsm = trans_to_cuda(lsm)
        targets = trans_to_cuda(targets)
        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class CenterLoss(nn.Module):
    """Centerloss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def get_center(self):
        return self.centers

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        labels = trans_to_cuda(labels)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

def TripletLoss(anchor, pos, neg, margin=1):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    Loss = nn.TripletMarginLoss(margin=margin, p=2)
    loss = Loss(anchor, pos, neg)
    return loss