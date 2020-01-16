import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PaperTextCNN(nn.Module):
    def __init__(self, hparams, embedding):
        super(PaperTextCNN, self).__init__()
        self.hparams = hparams
        self.embed = embedding
        self.conv1 = nn.Conv2d(1, hparams.txtcnn_pfilter_num1, (2, self.embed.embedding_dim))
        self.conv2 = nn.Conv2d(1, hparams.txtcnn_pfilter_num2, (3, self.embed.embedding_dim))
        self.conv3 = nn.Conv2d(1, hparams.txtcnn_pfilter_num3, (5, self.embed.embedding_dim))
        self.conv4 = nn.Conv2d(1, hparams.txtcnn_pfilter_num4, (7, self.embed.embedding_dim))
        
        # self.batch_norm = nn.BatchNorm1d(sum(kn for (kn, _) in kernel_settings))
        # self.fc = nn.Linear(sum(kn for (kn, _) in kernel_settings), output_dim)
        
        self.txtcnn_output_dim = hparams.txtcnn_pfilter_num1 + hparams.txtcnn_pfilter_num2 + hparams.txtcnn_pfilter_num3 + hparams.txtcnn_pfilter_num4
    
    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence, embed_dim)
        x = conv(x)
        # x: (batch, kn, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kn, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x: (batch, kn)
        return x
    
    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)
        x2 = self.conv_and_pool(x, self.conv2)
        x3 = self.conv_and_pool(x, self.conv3)
        x4 = self.conv_and_pool(x, self.conv4)
        x = torch.cat((x1, x2, x3, x4), 1)
        #x = F.dropout(x, p=self.hparams.txtcnn_drop_prob, training=self.training)
        # x = self.batch_norm(x)
        # x = self.fc(x)
        # logit = F.log_softmax(x, dim=1)
        return x
    
class RepoTextCNN(nn.Module):
    def __init__(self, hparams, embedding):
        super(RepoTextCNN, self).__init__()
        self.hparams = hparams
        self.embed = embedding
        self.conv1 = nn.Conv2d(1, hparams.txtcnn_rfilter_num1, (2, self.embed.embedding_dim))
        self.conv2 = nn.Conv2d(1, hparams.txtcnn_rfilter_num2, (4, self.embed.embedding_dim))
        
        self.txtcnn_output_dim = hparams.txtcnn_rfilter_num1 + hparams.txtcnn_rfilter_num2
    
    @staticmethod
    def conv_and_pool(x, conv):
        x = conv(x)
        x = F.relu(x.squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)
        x2 = self.conv_and_pool(x, self.conv2)
        x = torch.cat((x1, x2), 1)
        #x = F.dropout(x, p=self.hparams.txtcnn_drop_prob, training=self.training)
        return x
    
class TwoLayerGCN(nn.Module):
    def __init__(self, feature_dim, mid_dim, output_dim, gcn_dropout):
        super(TwoLayerGCN, self).__init__()
        self.gcn_dropout = gcn_dropout
        
        self.conv1 = GCNConv(feature_dim, mid_dim)
        self.conv2 = GCNConv(mid_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.gcn_dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class PaperModel(nn.Module):
    def __init__(self, hparams, data, embedding):
        super(PaperModel, self).__init__()
        self.hparams = hparams
        self.bridge_ids = data['bridge_ids']
        self.paper_edge_index = data['paper_edge_index']
        self.paper_features = data['paper_features']
        
        self.text_cnn = PaperTextCNN(hparams, embedding)
        self.batch_norm = nn.BatchNorm1d(self.text_cnn.txtcnn_output_dim)
        self.gcn2 = TwoLayerGCN(self.text_cnn.txtcnn_output_dim, 
                                hparams.gcn_mid_dim,
                                hparams.gcn_output_dim,
                                hparams.gcn_drop_prob)
    
    def forward(self, paper_index):
        all_features = self.text_cnn(self.paper_features)
        all_features = self.batch_norm(all_features)
        all_embeddings = F.normalize(self.gcn2(all_features, self.paper_edge_index), dim=1)
        
        bridges_paper_embeddings = all_embeddings.index_select(0, self.bridge_ids)
        selected_paper_embeddings = all_embeddings.index_select(0, paper_index)
        return bridges_paper_embeddings, selected_paper_embeddings
    
class RepoModel(nn.Module):
    def __init__(self, hparams, data, embedding):
        super(RepoModel, self).__init__()
        self.hparams = hparams
        self.bridge_ids = data['bridge_ids']
        self.repo_edge_index = data['repo_edge_index']
        self.repo_features = data['repo_features']
        self.repo_tags = data['repo_tags']
        
        self.gcn_output_dim = hparams.gcn_output_dim
        
        self.embed = embedding
        self.text_cnn = RepoTextCNN(hparams, embedding)
        self.batch_norm1 = nn.BatchNorm1d(self.text_cnn.txtcnn_output_dim)
        self.gcn2 = TwoLayerGCN(self.text_cnn.txtcnn_output_dim, 
                                hparams.gcn_mid_dim,
                                hparams.gcn_output_dim,
                                hparams.gcn_drop_prob)
        
        self.fc1 = nn.Sequential(
                        nn.Linear(self.embed.embedding_dim, hparams.txtcnn_rfilter_num1, bias=False),
                        nn.ReLU()
                    )
        self.fc2 = nn.Sequential(
                        nn.Linear(self.embed.embedding_dim, hparams.txtcnn_rfilter_num2, bias=False),
                        nn.ReLU()
                    )
        self.batch_norm2 = nn.BatchNorm1d(self.text_cnn.txtcnn_output_dim)
    
    def forward(self, repo_index):
        bn = repo_index.size()[0]
        
        all_desp_features = self.text_cnn(self.repo_features)
        all_desp_features = self.batch_norm1(all_desp_features)
        
        all_tag_features1 = self.fc1(self.embed(self.repo_tags).sum(axis=2)).sum(axis=1)
        all_tag_features2 = self.fc2(self.embed(self.repo_tags).sum(axis=2)).sum(axis=1)
        all_tag_features = torch.cat((all_tag_features1, all_tag_features2), 1)
        all_tag_features = self.batch_norm2(all_tag_features)
        
        all_embeddings = F.normalize(self.gcn2(all_desp_features + all_tag_features, self.repo_edge_index), dim=1)
        
        bridges_repo_embeddings = all_embeddings.index_select(0, self.bridge_ids)
        selected_repo_embeddings = all_embeddings.index_select(0, repo_index.view(-1))
        selected_repo_embeddings = selected_repo_embeddings.view(bn, -1, self.gcn_output_dim)
        
        return bridges_repo_embeddings, selected_repo_embeddings
    
class P2r(nn.Module):
    def __init__(self, hparams, data):
        super(P2r, self).__init__()
        self.embed = nn.Embedding.from_pretrained(data['word_embeddings'], freeze=hparams.freeze_embeddings)
        self.paperModel = PaperModel(hparams, data, self.embed)
        self.repoModel = RepoModel(hparams, data, self.embed)
    
    def forward(self, paper_index, repo_index):
        bridges_paper_embeddings, selected_paper_embeddings = self.paperModel(paper_index)
        bridges_repo_embeddings, selected_repo_embeddings = self.repoModel(repo_index)
        
        constraint = 1 - torch.sum(bridges_paper_embeddings * bridges_repo_embeddings, axis=1)
        scores = torch.sum(selected_paper_embeddings.unsqueeze(1) * selected_repo_embeddings, axis=2)
        ranks = torch.argsort(torch.argsort(scores, descending=True)) + 1
        
        return constraint, scores, ranks

