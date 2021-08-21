import numpy as np
import torch
from torch.utils.data import Dataset
import Utils

class Data(Dataset):
    def __init__(
        self,
        start_pdbs,
        seq_feat_types = ['onehot', 'rPosition', 'SepEnc'], 
        struc_feat_types = ['SS3', 'RSA', 'Dihedral', 'Ca1-Ca2', 'Cb1-Cb2', 'N1-O2', 'Ca1-Cb1-Cb2', 'N1-Ca1-Cb1-Cb2', 'Ca1-Cb1-Cb2-Ca2'], 
        adj_type = 'Cb1-Cb2',
        adj_cutoff = 10,
    ):
        super().__init__()
        self.start_pdbs = start_pdbs
        self.seq_feat_types = seq_feat_types
        self.struc_feat_types = struc_feat_types
        self.adj_type = adj_type
        self.adj_cutoff = adj_cutoff

        self.feat_class = {
            'seq':{
                'node':['onehot', 'rPosition'],
                'edge':['SeqEnc'],
            },
            'struc':{
                'node':['SS3', 'RSA', 'Dihedral'],
                'edge':['Ca1-Ca2', 'Cb1-Cb2', 'N1-O2', 'Ca1-Cb1-Cb2', 'N1-Ca1-Cb1-Cb2', 'Ca1-Cb1-Cb2-Ca2'],
            },
        }
        self.data_len = len(self.start_pdbs)
    
    def __len__(self):
        return self.data_len
    
    def __get_seq_feature(self, pdb_file):
        """
        Args:
            pdb_file: str. File path
        Return:
            node_feat: dict.
            edge_feat: dict.
            len(seq): int.
        """
        seq = Utils.get_seqs_from_pdb(pdb_file)
        # node feat
        node_feat = {
            'onehot': Utils.get_seq_onehot(seq),
            'rPosition': Utils.get_rPos(seq)
        }
        # edge_feat:
        edge_feat = {
            'SeqEnc': Utils.get_SeqEnc(seq)
        }
        return node_feat, edge_feat, len(seq)
    
    def __get_struc_feature(self, pdb_file, seq_len):
        # node feat
        node_feat = Utils.get_DSSP_label(pdb_file, [1,seq_len])
        # atom embedding
        embedding = Utils.get_atom_emb(pdb_file, [1,seq_len])
        node_feat['atom_emb'] = {
            'embedding': embedding.astype(np.float32)
        }
        # edge feat
        edge_feat = Utils.calc_geometry_maps(pdb_file, [1,seq_len], geometry_types=self.feat_class['struc']['edge'])
        return node_feat, edge_feat
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        pdb_file = self.start_pdbs[idx]
        print(pdb_file, 'gen feature ...')

        feature = {'node':None, 'edge':None}

        # seq feature
        seq_node_feat, seq_edge_feat, seq_len = self.__get_seq_feature(pdb_file)
        for _feat in self.feat_class['seq']['node']:
            feature['node'] = seq_node_feat[_feat] if feature['node'] is None else np.concatenate([feature['node'], seq_node_feat[_feat]], axis=-1)
        for _feat in self.feat_class['seq']['edge']:
            feature['edge'] = seq_edge_feat[_feat] if feature['edge'] is None else np.concatenate([feature['edge'], seq_edge_feat[_feat]], axis=-1)
        
        # struc feature
        struc_node_feat, struc_edge_feat = self.__get_struc_feature(pdb_file, seq_len)
        for _feat in self.feat_class['struc']['node']:
            feature['node'] = struc_node_feat[_feat] if feature['node'] is None else np.concatenate([feature['node'], struc_node_feat[_feat]], axis=-1)
        for _feat in self.feat_class['struc']['edge']:
            feature['edge'] = struc_edge_feat[_feat] if feature['edge'] is None else np.concatenate([feature['edge'], struc_edge_feat[_feat]], axis=-1)
        
        # adj
        adj_data = struc_edge_feat[self.adj_type]*10
        adj_data = np.squeeze(adj_data, axis=2)
        adj_data[adj_data<0] = 10000
        adj = np.zeros_like(adj_data)
        adj[adj_data <= self.adj_cutoff] = 1

        # feature
        feature = np.nan_to_num(feature)
        feature['node'] = feature['node'].astype(np.float32) # (N, 28)
        feature['feat'] = feature['node'].astype(np.float32) # (N, N, 15)
        feature['adj'] = feature['adj'].astype(np.float32)   # (N, N)
        feature['atom_emb'] = struc_node_feat['atom_emb']

        # sample
        sample = {
            'feature': feature,
            'pdb_info': {'pdb': pdb_file},
        }

        return sample