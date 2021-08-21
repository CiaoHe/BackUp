from __future__ import print_function, division
import os, pickle, tempfile
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader
import dgl
import argparse

from Data import Data
from Model import GNN

__MYPATH__ = os.path.split(os.path.realpath(__file__))[0]

class GNNPred:
    def __init__(self, device_id, distCB=True, QA=False) -> None:
        """
        Args:
            device_id: int. the GPU id, -1 means CPU
            distCB: Bool. GNN model for refine distance prediction
            QA:  Bool. GNN model for quality assessment task
        """
        self.distCB, self.QA = distCB, QA
        assert (not distCB == QA), "please select only one task: distCB or QA."
        self.device = torch.device('cuda:%s'%(device_id)) if device_id>=0 else torch.device('cpu')
        self.model = GNN(distCB=distCB, QA=QA).to(self.device)
    
    def load_model(self, params_file):
        print('load model params.')
        model_dict = torch.load(params_file, map_location=self.device)
        self.model.load_state_dict(model_dict)
    
    def forward(self, sample):
        # build graph
        node_feat, edge_feat, adj = sample['feature']['node'].squeeze(0), sample['feature']['edge'].squeeze(0), \
            sample['feature']['adj'].squeeze(0) # remove batch_dim=1 
        graph = dgl.DGLGraph()
        graph.add_nodes(node_feat.shape[0])
        ii,jj = np.where(adj==1)
        graph.add_edges(ii,jj)
        graph.ndata['nfeat'] = node_feat
        graph.edata['efeat'] = edge_feat[ii,jj]

        # atom_emb
        atom_emb = sample['feature']['atom_emb']
        graph.ndata['atom_emb'] = atom_emb['embedding'].squeeze(0) # (N, 14, 7)

        # forward
        output = self.model(graph.to(self.device))

        # result
        pred = {'pdb':sample['pdb_info']['pdb'][0], }
        if self.distCB:
            pred['adj_pair'] = torch.stack([graph.all_edges()[0], graph.all_edges()[1]], dim=-1).cpu().numpy()
            pred['distCB'] = F.softmax(output['distCB'], dim=-1).cpu().numpy()
        if self.QA:
            pred['global_lddt'] = float(output['global_lddt'].cpu().numpy())
            pred['local_lddt'] = output['local_lddt'].squeeze(1).cpu().numpy().tolist()
        
        return pred
    
    def pred(self, data_loader):
        print('Run pred')
        self.model.eval()
        with torch.no_grad():
            results = []
            for sample in data_loader:
                pdb_info = self.forward(sample)
                results.append(pdb_info)
                print(pdb_info['pdb'], 'done.')
            return results

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="path of starting model or folder containing starting models.")
    parser.add_argument("output", type=str, help="path of output folder.")

    parser.add_argument('-n_decoy', type=int, dest='n_decoy', default=1, help='number of decoys built in each iteration.')
    parser.add_argument('-n_proc', type=int, dest='n_proc', default=1, help='number of processes running in parallel (>=1, recommendation is >=n_decoy).')
    parser.add_argument('-device_id', type=int, dest='device_id', default=-1, help='device id (-1 for CPU, >=0 for GPU).')

    parser.add_argument("-save_qa", action="store_true", default=False, help="save QA results.")
    parser.add_argument("-save_le_decoy", action="store_true", default=False, help="save decoy models of lowest erergy in each iteration.")
    parser.add_argument("-save_all_decoy", action="store_true", default=False, help="save all decoy models.")
    parser.add_argument("-only_pred_dist", action="store_true", default=False, help="only predict the refined distance probability distribution.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)

    # set the CUDA_VISIBLE_DEVICES
    print("Using devices", args.device_id)
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    if args.device_id>=0: args.device_id = 0

    assert args.n_proc >= 1, 'n_proc must be >=1 '
    torch.set_num_threads(args.n_proc)

    if not args.only_pred_dist: from Folding import folding

    # process pdb
    start_pdbs = []
    if os.path.isfile(args.input):
        start_pdbs = [args.input,]
    elif os.path.isdir(args.input):
        for pdb in os.listdir(args.input):
            start_pdbs.append(os.path.join(args.input, pdb))
    assert len(start_pdbs)>0, "start model not found."
    if not os.path.isdir(args.output): os.mkdir(args.output)

    print('refinement')
    gnn_param_dir = '../data/gnn_params/'
    gnn_params = ['model.1.pkl', 'model.2.pkl', 'model.3.pkl', 'model.DAN1.pkl', 'model.DAN2.pkl',]
    tmp_dir = tempfile.TemporaryDirectory(dir = '/tmp/')
    _pdbs, lowenergy_pdbs = start_pdbs.copy(), {}
    gnn_pred = GNNPred(args.device_id, distCB=True, QA=False)
    for gnn_param in gnn_params:
        print(gnn_param, 'start')
        # work_dir
        work_dir = "%s/%s/"%(tmp_dir.name, gnn_param.split('.pkl')[0])
        if not os.path.isdir(work_dir): os.mkdir(work_dir)
        # dataset
        data = Data(_pdbs)
        data_loader = DataLoader(data, pin_memory=True, num_workers=args.n_proc)
        # pred
        gnn_pred.load_model(gnn_param_dir + gnn_param)
        refined_dist = gnn_pred.pred(data_loader)
        # refined dist
        if args.only_pred_dist:
            refined_dist_file = "%s/refined_dist.%s.pkl"%(args.output, gnn_param.split('.pkl')[0])
            pickle.dump(refined_dist, open(refined_dist_file, 'wb'))
            print("refined dist saved at %s"%(refined_dist_file))
            continue
        else:
            refined_dist_file = "%s/refined_dist.pkl"%(work_dir)
            pickle.dump(refined_dist, open(refined_dist_file, "wb"))
        
        # folding and select low energy pdb
        _lowenergy_pdbs:dict = folding(refined_dist, work_dir, n_decoy=args.n_decoy, n_proc=args.n_proc)
        lowenergy_pdbs[gnn_param] = _lowenergy_pdbs
        _pdbs = [_lowenergy_pdbs[_] for _ in _lowenergy_pdbs]
        print(_pdbs)
        print(gnn_param, 'done')
    
    if args.only_pred_dist:
        exit()
    
    print('QA')
    qa_gnn_param = 'model.QA.pkl'
    gnn_QA:nn.Module = GNNPred(args.device_id, distCB=False, QA=True)
    gnn_QA.load_model(gnn_param_dir+qa_gnn_param)
    for start_pdb in start_pdbs:
        le_pdbs, _pdb = [], start_pdb
        for gnn_param in gnn_params:
            _pdb = lowenergy_pdbs[gnn_param][_pdb]
            le_pdbs.append(_pdb)
        # QA
        data = Data(le_pdbs)
        data_loader = DataLoader(data, pin_memory=True, num_workers=args.n_proc)
        qa_results = gnn_QA.pred(data_loader)
        # select pdb by QA (choose top one)
        selected_item = sorted(qa_results, key=lambda k: -k['global_lddt'])[0]
        refined_pdb = "%s/%s.refined.pdb"%(args.output, start_pdb.split('/')[-1])
        os.system("cp %s %s"%(selected_item['pdb'], refined_pdb))
        # save QA results
        if args.save_qa:
            selected_item['pdb'] = refined_pdb
            pickle.dump(selected_item, open("%s.qa.pkl"%(refined_pdb)), "wb")
        # save low energy decoy models
        if args.save_le_decoy:
            decoys_dir = "%s/lowenergy_decoys/"%(args.output)
            if not os.path.isdir(decoys_dir): os.mkdir(decoys_dir)
            os.system("cp %s %s/" % (' '.join(le_pdbs), decoys_dir))
            print("low energy decoys saved at %s"%(decoys_dir))
    
    # save all decoy models
    if args.save_all_decoy:
        decoys_dir = "%s/all_decoys/"%(args.output)
        if not os.path.isdir(decoys_dir): os.mkdir(decoys_dir)
        os.system("cp %s/*/*.pdb %s"%(tmp_dir.name, decoys_dir))
        print("all decoys saved at %s"%(decoys_dir))

    tmp_dir.cleanup()
    print("all done.")