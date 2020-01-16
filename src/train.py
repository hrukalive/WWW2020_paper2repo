import os
import sys
import pickle as pkl

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import Experiment, HyperOptArgumentParser

from p2r_module import P2rSystem

def main(hparams, data):
    # init experiment
    log_dir = os.path.dirname(os.path.realpath(__file__))
    exp = Experiment(
        name=hparams.exp_name,
        debug=False,
        save_dir=log_dir,
        version=0,
        autosave=True,
        description='P2R codebase'
    )

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = P2rSystem(hparams, data)

    model_save_path = '{}/{}/version_{}/checkpoints'.format(exp.save_dir, exp.name, exp.version)
    checkpoint = ModelCheckpoint(filepath=model_save_path, verbose=True, monitor='tng_loss', mode='min', save_best_only=True)

    # configure trainer
    trainer = Trainer(
        experiment = exp,
        checkpoint_callback = checkpoint,
        min_nb_epochs = 1,
        max_nb_epochs = hparams.max_nb_epochs,
        track_grad_norm = 2,
        accumulate_grad_batches=1,
        row_log_interval=1,
        amp_level='O2',
        use_amp=True,
        gpus=1
    )

    # train model
    trainer.fit(model)
    trainer.test()
    
    filepath = '{}/_ckpt_epoch_final.ckpt'.format(model_save_path)
    checkpoint.save_model(filepath, False)

def loadData(data_root, train_div):
    with open('%s/ind.paper-repo.data' % data_root, 'rb') as f:
        content = pkl.load(f)

    paper_edge_index = []
    for idx, edges in enumerate(content['paperGraphAdjList']):
        for edge in sorted(list(edges)):
            paper_edge_index.append([idx, edge])
    repo_edge_index = []
    for idx, edges in enumerate(content['repoGraphAdjList']):
        for edge in sorted(list(edges)):
            repo_edge_index.append([idx, edge])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    p2r_data = {
        'paper_graph_adjlist': content['paperGraphAdjList'],
        'paper_edge_index': torch.LongTensor(paper_edge_index).t().contiguous().to(device),
        'paper_features': torch.LongTensor(content['paperFeatures']).to(device),
        'cofork_repo_graph_adjlist': content['coforkRepoGraphAdjList'],
        'repo_graph_adjlist': content['repoGraphAdjList'],
        'repo_edge_index': torch.LongTensor(repo_edge_index).t().contiguous().to(device),
        'repo_features': torch.LongTensor(content['repoFeatures']).to(device),
        'repo_tags': torch.LongTensor(content['repoTags']).to(device),
        'positives': content['positives'],
        'bridge_length': int(content['bridgeLength'] * train_div),
        'bridge_ids': torch.LongTensor(list(filter(lambda x: x < int(content['bridgeLength'] * train_div), content['bridgeIds']))).to(device),
        'word_embeddings': torch.FloatTensor(content['wordEmbeddings']).to(device)
    }

    print('Training div: {} BLength {} {}'.format(train_div, p2r_data['bridge_length'], len(p2r_data['bridge_ids'].tolist())))
    return p2r_data
    
if __name__ == '__main__':

    # use default args given by lightning
    root_dir = os.path.dirname(os.path.realpath(sys.modules['__main__'].__file__))
    parent_parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
    parent_parser.add_argument('--root_dir', default=root_dir, type=str)
    parent_parser.add_argument('--exp_name', default='p2r_experiment', type=str)

    # allow model to overwrite or extend args
    parser = P2rSystem.add_model_specific_args(parent_parser, root_dir)
    hparams = parser.parse_args()
    
    p2r_data = loadData(hparams.data_root, hparams.train_div)

    # train model
    # for trial_hparams in hparams.trials(2):
    main(hparams, p2r_data)
