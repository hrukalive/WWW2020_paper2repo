import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

def warpLoss(scores, neg_split, margin, device):
    err = torch.tensor(0.0).to(device)
    for b in range(scores.size(0)):
        pos = scores[b][:neg_split[b]]
        neg = scores[b][neg_split[b]:]
        discrim = margin - pos.repeat(neg.size(0), 1).t() + neg.repeat(pos.size(0), 1)
        rank = discrim.gt(0).sum(axis=1).float().unsqueeze(1)
        score = F.relu(discrim).div(rank)
        score[score != score] = 0
        err += (torch.log1p(rank) * score.sum(axis=1).unsqueeze(1)).sum()
    return err

def metricMAP(ranks, neg_split, k = 20):
    filtered = [sorted(filter(lambda x: x <= k, rank[:s])) for rank, s in zip(ranks.tolist(), neg_split.tolist())]
    return np.mean(list(map(lambda rank: np.mean([(i + 1) / r for i, r in enumerate(rank)]) if len(rank) > 0 else 0, filtered)))

def metricMRR(ranks, neg_split, k = 20):
    filtered = [sorted(filter(lambda x: x <= k, rank[:s])) for rank, s in zip(ranks.tolist(), neg_split.tolist())]
    return np.mean(list(map(lambda rank: 1.0 / rank[0] if len(rank) > 0 else 0, filtered)))

def metricAccuracy(ranks, neg_split, k = 20):
    extracted = [sorted(rank[:s]) for rank, s in zip(ranks.tolist(), neg_split.tolist())]
    return np.mean(list(map(lambda rank: len(list(filter(lambda x: x <= k, rank))) / len(rank), extracted)))

def metricPMAP(ranks, neg_split, k = 20):
    return np.sqrt(metricMAP(ranks, neg_split, k) * metricAccuracy(ranks, neg_split, k))
