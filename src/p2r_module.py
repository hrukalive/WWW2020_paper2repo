import os
import sys
import itertools
import random
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import data_loader, LightningModule
from test_tube import Experiment, HyperOptArgumentParser

from models import P2r
from metrics import warpLoss, metricMAP, metricMRR, metricAccuracy, metricPMAP
from dataset import P2rTrainDataset, P2rTestDataset

class P2rSystem(LightningModule):
    def __init__(self, hparams, data):
        super(P2rSystem, self).__init__()
        
        self.hparams = hparams
        self.data = data
        self.device = hparams.device
        print(self.device)
        
        self.__build_dataset()
        self.__build_model()
        
    def __build_model(self):
        self.p2r = P2r(self.hparams, self.data)
        self.p2r.to(self.device)
    
    def __build_dataset(self):
        random.seed(42)
        bridge_paper_indices = set(range(self.data['bridge_length'])) - set(self.data['positives'].keys())
        all_paper_indices = list(OrderedDict.fromkeys(list(bridge_paper_indices) + list(itertools.chain(*[self.data['paper_graph_adjlist'][idx] for idx in bridge_paper_indices]))).keys())
        for i in range(len(all_paper_indices) - 1, -1, -1):
            if all_paper_indices[i] in self.data['positives']:
                all_paper_indices.pop(i)
        train_paper_indices = all_paper_indices
        random.seed()
        
        self.p2r_train_dataset = P2rTrainDataset(self.hparams, self.data, train_paper_indices)
        self.p2r_test_dataset = P2rTestDataset(self.hparams, self.data, self.data['positives'])
        
        self.p2r_train_loader = DataLoader(dataset=self.p2r_train_dataset, 
                                           batch_size=self.hparams.batch_size, 
                                           shuffle=self.hparams.shuffle)
        self.p2r_test_loader = DataLoader(dataset=self.p2r_test_dataset, 
                                          batch_size=len(self.p2r_test_dataset), 
                                          shuffle=self.hparams.shuffle)

    def forward(self, paper_index, repo_index):
        return self.p2r(paper_index, repo_index)
    
    def loss(self, scores, neg_split, margin, constraint):
        return warpLoss(scores, neg_split, margin, self.device) * (1 + constraint.mean() / 2)

    def __one_step(self, batch, batch_nb):
        paper_index, repo_indices, neg_split = batch
        constraint, scores, ranks = self.forward(paper_index, repo_indices)
        loss = self.loss(scores, neg_split, self.hparams.warploss_margin, constraint)
        m_maps = [('MAP@%d' % k, torch.tensor(metricMAP(ranks, neg_split, k))) for k in [5, 10, 15, 20]]
        m_mrrs = [('MRR@%d' % k, torch.tensor(metricMRR(ranks, neg_split, k))) for k in [5, 10, 15, 20]]
        m_accs = [('ACC@%d' % k, torch.tensor(metricAccuracy(ranks, neg_split, k))) for k in [5, 10, 15, 20]]
        m_pmaps = [('PMAP@%d' % k, torch.tensor(metricPMAP(ranks, neg_split, k))) for k in [5, 10, 15, 20]]
        return loss, list(itertools.chain(*[m_maps, m_mrrs, m_accs, m_pmaps]))
    
    def training_step(self, batch, batch_nb):
        loss_val, metrics = self.__one_step(batch, batch_nb)
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
        ret = OrderedDict([('loss', loss_val), ('progress', OrderedDict([('tng_loss', loss_val)] + metrics))])
        return ret
    
    def test_step(self, batch, batch_nb):
        loss, metrics = self.__one_step(batch, batch_nb)
        ret = OrderedDict([('test_loss', loss)] + list(metrics))
        return ret
    
    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        vals = list(itertools.chain(*[ \
            [('test_avg_{0:s}@{1:d}'.format(metric, k), \
                    torch.stack([x['{0:s}@{1:d}'.format(metric, k)] for x in outputs]).mean()) \
                for k in [5, 10, 15, 20]] \
            for metric in ['MAP', 'MRR', 'ACC', 'PMAP'] \
        ]))
        ret = OrderedDict([('avg_test_loss', avg_loss)] + vals)
        return ret

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # scheduler = scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return optimizer #], [scheduler]

    @data_loader
    def train_dataloader(self):
        return self.p2r_train_loader

    @data_loader
    def test_dataloader(self):
        return self.p2r_test_loader
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])
        
        parser.set_defaults(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # network params
        parser.opt_list('--gcn_mid_dim', default=256, type=int, options=[128, 256, 512, 1024], tunable=True)
        parser.opt_list('--gcn_output_dim', default=256, type=int, options=[128, 256, 512, 1024], tunable=True)
        parser.opt_list('--txtcnn_drop_prob', default=0.0, options=[0.0, 0.1, 0.2], type=float, tunable=True)
        parser.opt_list('--gcn_drop_prob', default=0.5, options=[0.2, 0.5], type=float, tunable=True)
        parser.opt_list('--warploss_margin', default=0.4, type=float, tunable=True)
        parser.opt_list('--freeze_embeddings', default=True, options=[True, False], 
                        type=lambda x: (str(x).lower() == 'true'), tunable=True)
        
        parser.opt_list('--txtcnn_pfilter_num1', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_pfilter_num2', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_pfilter_num3', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_pfilter_num4', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_rfilter_num1', default=64, options=[16, 32, 64, 128], type=int, tunable=True)
        parser.opt_list('--txtcnn_rfilter_num2', default=32, options=[16, 32, 64, 128], type=int, tunable=True)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'data'), type=str)
        parser.add_argument('--top_t', default=6, type=int)
        parser.add_argument('--total_onehop', default=20, type=int)
        parser.add_argument('--total', default=50, type=int)
        parser.add_argument('--shuffle', default=True, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--train_div', default=1.0, type=float)

        # training params (opt)
        parser.opt_list('--batch_size', default=64, options=[32, 64, 128, 256], type=int, tunable=False)
        parser.opt_list('--max_nb_epochs', default=8, options=[256, 512, 1024], type=int, tunable=False)
        parser.opt_list('--learning_rate', default=0.0005, options=[0.0001, 0.0005, 0.001], type=float, tunable=True)
        parser.opt_list('--weight_decay', default=0.001, options=[0.0001, 0.0005, 0.001], type=float, tunable=True)
        parser.add_argument('--model_save_path', default=os.path.join(root_dir, 'experiment'), type=str)
        return parser
    
