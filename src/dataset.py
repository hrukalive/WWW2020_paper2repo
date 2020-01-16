import itertools
import random

import torch
from torch.utils.data.dataset import Dataset

class P2rTrainDataset(Dataset):
    def __init__(self, hparams, data, paper_indices):
        super(P2rTrainDataset, self).__init__()
        self.paper_indices = paper_indices
        self.top_t = hparams.top_t
        self.total_onehop = hparams.total_onehop
        self.total = hparams.total
        self.bridge_length = data['bridge_length']
        self.bridge_ids = set(data['bridge_ids'].tolist())
        self.paper_graph_adjlist = data['paper_graph_adjlist']
        self.cofork_repo_graph_adjlist = data['cofork_repo_graph_adjlist']
        self.repo_graph_adjlist = data['repo_graph_adjlist']
        self.positives = {k: set(v) for k, v in data['positives'].items()}
    
    def __getitem__(self, index):
        paper_index = self.paper_indices[index]
        if paper_index < self.bridge_length:
            pos_candidate = set(([paper_index] + self.cofork_repo_graph_adjlist[paper_index])[:self.top_t])
            neg_onehop_candidate = set(itertools.chain(*[self.repo_graph_adjlist[idx] for idx in pos_candidate]))
            neg_onehop_candidate -= pos_candidate
            neg_others = set(range(len(self.repo_graph_adjlist))) - pos_candidate - neg_onehop_candidate
            pos_candidate = list(pos_candidate)
        else:
            pos_candidate = list(filter(lambda x: x in self.bridge_ids, self.paper_graph_adjlist[paper_index]))
            neg_onehop_candidate = set(itertools.chain(*[self.repo_graph_adjlist[idx] for idx in pos_candidate]))
            neg_onehop_candidate -= set(pos_candidate)
            neg_others = set(range(len(self.repo_graph_adjlist))) - set(pos_candidate) - neg_onehop_candidate
            pos_candidate = set(pos_candidate)
            pos_candidate = random.sample(list(pos_candidate), min(len(pos_candidate), self.top_t))
        
        neg_split = len(pos_candidate)
        neg_others_split = self.total_onehop - neg_split
        neg_onehop_candidate = random.sample(list(neg_onehop_candidate), min(len(neg_onehop_candidate), neg_others_split))
        neg_rest = self.total - neg_split - len(neg_onehop_candidate)
        repo_indices = torch.LongTensor(list(itertools.chain(*[
            sorted(pos_candidate), 
            sorted(neg_onehop_candidate), 
            sorted(random.sample(list(neg_others), min(len(neg_others), neg_rest)))
        ])))
        return paper_index, repo_indices, neg_split

    def __len__(self):
        return len(self.paper_indices)

class P2rTestDataset(Dataset):
    def __init__(self, hparams, data, positives):
        super(P2rTestDataset, self).__init__()
        self.bridge_length = data['bridge_length']
        self.positives = positives
        for k, v in self.positives.items():
            if k < self.bridge_length and k not in v:
                v.append(k)
        self.positives_key = sorted(list(self.positives.keys()))
        self.top_t = hparams.top_t
        self.total_onehop = hparams.total_onehop
        self.total = hparams.total
        self.paper_graph_adjlist = data['paper_graph_adjlist']
        self.repo_graph_adjlist = data['repo_graph_adjlist']
    
    def __getitem__(self, index):
        paper_index = self.positives_key[index]
        pos_candidate = set(self.positives[paper_index])
        if paper_index < self.bridge_length:
            neg_onehop_candidate = set(itertools.chain(*[self.repo_graph_adjlist[idx] for idx in pos_candidate]))
            neg_onehop_candidate -= pos_candidate
        else:
            neg_onehop_candidate = list(filter(lambda x: x < self.bridge_length, self.paper_graph_adjlist[paper_index]))
            neg_onehop_candidate += list(itertools.chain(*[self.repo_graph_adjlist[idx] for idx in neg_onehop_candidate]))
            neg_onehop_candidate = set(neg_onehop_candidate) - pos_candidate
        neg_others = set(range(len(self.repo_graph_adjlist))) - pos_candidate - neg_onehop_candidate
        
        pos_candidate = random.sample(list(pos_candidate), min(self.top_t, len(pos_candidate)))
        neg_split = len(pos_candidate)
        neg_others_split = self.total_onehop - neg_split
        neg_onehop_candidate = random.sample(list(neg_onehop_candidate), min(len(neg_onehop_candidate), neg_others_split))
        neg_rest = self.total - neg_split - len(neg_onehop_candidate)
        repo_indices = torch.LongTensor(list(itertools.chain(*[
            sorted(pos_candidate), 
            sorted(neg_onehop_candidate), 
            sorted(random.sample(list(neg_others), min(len(neg_others), neg_rest)))
        ])))
        return paper_index, repo_indices, neg_split

    def __len__(self):
        return len(self.positives)
