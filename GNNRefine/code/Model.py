from dgl._deprecate.graph import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.udf import EdgeBatch, NodeBatch

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
    ])

def norm_func(norm, n_channel):
    return nn.ModuleDict([
        ['instance', nn.InstanceNorm1d(n_channel)],
    ])[norm]

class AtomEmbLayer(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        *args,
        atom_emb_in = 7,  # atom_emb dim = 7
        atom_emb_h = 256, # hidden dim
        norm='instance',
        activation = 'relu',
        **kwargs
    ):
        super().__init__()
        self.norm = norm

        self.fn_atom_norm = norm_func(self.norm, atom_emb_in)
        self.fn_atom_layer = nn.Linear(atom_emb_in, atom_emb_h, bias=False)
        self.fn_atom_activation = activation_func(activation)
        self.fn_atom_norm2 = norm_func(self.norm, atom_emb_h)
        self.fn_atom_layer2 = nn.Linear(atom_emb_h, atom_emb_h, bias=False)
    
    def forward(self, G:dgl.DGLGraph):
        atom_emb = G.ndata['atom_emb'] # (N, 14, 7)
        # first layer
        atom_emb = self.fn_atom_norm(atom_emb)
        atom_emb = self.fn_atom_layer(atom_emb)
        atom_emb = torch.mean(atom_emb, dim = 1, keepdim=True) # (N, 1, 256)
        atom_emb = self.fn_atom_activation(atom_emb)
        # second layer
        atom_emb = self.fn_atom_norm2(atom_emb).squeeze() if self.norm == 'instance' else \
            self.fn_atom_norm2(atom_emb.squeeze()) # (N, 256)
        atom_emb = self.fn_atom_layer2(atom_emb)
        atom_emb = self.fn_atom_activation(atom_emb)
        # cat
        x = torch.cat((G.ndata['nfeat'], atom_emb), dim = -1)
        G.ndata['feat'] = x

        return G

class EdgeApplyModule(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        *args,
        norm = 'instance',
        activation = 'leaky_relu',
        LSTM = False,
        **kwargs
    ):
        super().__init__()
        self.norm = norm
        self.LSTM = LSTM
        
        self.fn_norm = norm_func(self.norm, n_in)
        self.fn_linear = nn.Linear(n_in, n_out)
        self.fn_activation = activation_func(activation)
        if self.LSTM: self.fn_lstm = nn.LSTMCell(n_out, n_out, bias=False)
        self.attn_fc = nn.Linear(n_out, 1, bias=False)

    def forward(self, edges:EdgeBatch):
        x = torch.cat([edges.src['nfeat'], edges.data['efeat'], edges.dst['nfeat']], dim = 1)
        x = self.fn_norm(x.unsqueeze(1)).squeeze() if self.norm == 'instance' else self.fn_norm(x)
        x = self.fn_linear(x)
        x = self.fn_activation(x)
        if self.LSTM:
            # hidden: efeat, 
            # cell: efeat_c
            if not 'efeat_c' in edges.data:
                edges.data['efeat_c'] = torch.zeros_like(edges.data['efeat'])
            x, c = self.fn_lstm(x, (edges.data['efeat'], edges.data['efeat_c']))
        
        attn = self.attn_fc(x)
        if self.LSTM:
            return {'efeat':x, 'attn':attn, 'efeat_c': c}
        else:
            return {'efeat':x, 'attn':attn, }

def message_func(edge):
    return {'_efeat':edge.data['efeat'], '_attn':edge.data['attn']}

def reduce_func(node:NodeBatch):
    alpha = F.softmax(node.mailbox['_attn'], dim=1)
    attn_feat = torch.sum(alpha * node.mailbox['_efeat'], dim=1)
    feat = torch.cat([node.data['_efeat'], attn_feat], dim=1) # ?
    return {'_nfeat':feat}

class NodeApplyModule(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        *args,
        norm = 'instance',
        activation = 'leaky_relu',
        LSTM = False,
        **kwargs
    ):
        super().__init__()
        self.norm = norm
        self.LSTM = LSTM
        
        self.fn_norm = norm_func(self.norm, n_in)
        self.fn_linear = nn.Linear(n_in, n_out)
        self.fn_activation = activation_func(activation)
        if self.LSTM: self.fn_lstm = nn.LSTMCell(n_out, n_out, bias=False)
    
    def forward(self, nodes:NodeBatch):
        x = nodes.data['_nfeat']
        x = self.fn_norm(x.unsqueeze(1)).squeeze() if self.norm == 'instance' else self.fn_norm(x)
        x = self.fn_linear(x)
        x = self.fn_activation(x)
        if self.LSTM:
            # hidden: efeat, 
            # cell: efeat_c
            if not 'nfeat_c' in nodes.data:
                nodes.data['nfeat_c'] = torch.zeros_like(nodes.data['nfeat'])
            x, c = self.fn_lstm(x, (nodes.data['nfeat'], nodes.data['nfeat_c']))
            return {'nfeat':x, 'nfeat_c':c}
        return {'nfeat':x}

class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        node_n_in,  # node dim of in
        node_n_out, # node dim of out
        edge_n_in,  # edge dim of in
        edge_n_out, # edge dim of out
        norm = 'instance',
        activation = 'leaky_relu',
        LSTM = False,
        last_layer = False,
        atom_emb = False,
        atom_emb_h = 256, # hidden dimension of atom-embedding
        *args,
        **kwargs
    ):
        super().__init__()
        # Flags
        self.LSTM, self.last_layer = LSTM, last_layer

        self.edge_updator = EdgeApplyModule(edge_n_in + node_n_in*2, edge_n_out, norm=self.norm, LSTM=self.LSTM)
        self.node_updator = NodeApplyModule(node_n_in + edge_n_out, node_n_out, norm=self.norm, activation=activation,\
            LSTM=self.LSTM)
        
    def forward(self, G:DGLGraph):
        G.apply_edges(self.edge_updator)
        G.update_all(
            message_func=message_func, 
            reduce_func=reduce_func, 
            apply_node_func=self.node_updator
        )
        if self.last_layer:
            G.apply_nodes(self.node_updator)
        return G

class GNN(nn.Module):
    def __init__(
        self,
        node_n_in = 28,
        node_n_hidden = 256,
        edge_n_in = 15,
        edge_n_hidden = 256,
        n_layers = 10,
        n_output = 37,
        LSTM = True,
        atom_emb_in = 7,
        atom_emb_h = 256,
        norm = 'instance',
        activation = 'relu',
        distCB = True,
        QA = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.distCB = distCB
        self.QA = QA

        self.layers = nn.ModuleList([])
        # AtomEmbedder
        self.layers.append(AtomEmbLayer(node_n_in, node_n_hidden, atom_emb_in=atom_emb_in, \
            atom_emb_h=atom_emb_h, norm=norm, activation=activation))
        # First MessagePassingLayer
        self.layers.append(MessagePassingLayer(node_n_in+atom_emb_h, node_n_hidden, edge_n_in, edge_n_hidden, \
            norm=norm, activation=activation, LSTM=False, last_layer=False))
        # intermediate MessagePassingLayers
        for _ in range(n_layers-2):
            self.layers.append(MessagePassingLayer(node_n_hidden, node_n_hidden, edge_n_hidden, edge_n_hidden, \
                norm=norm, activation=activation, LSTM=True))
        # Last MessagePassingLayer
        self.layers.append(MessagePassingLayer(node_n_hidden, node_n_hidden, edge_n_hidden, edge_n_hidden, \
            norm=norm, activation=activation, LSTM=False, last_layer=True))
        
        # disctCB Layer
        if self.distCB:
            self.output_layer = nn.Sequential(
                nn.Linear(edge_n_hidden, n_output)
            )
        
        # QA Layer
        if self.QA:
            self.global_qa_linear = nn.Linear(node_n_hidden + edge_n_hidden, 1)
            self.local_qa_linear = nn.Linear(node_n_hidden, 1)
            self.sigmoid = nn.Sigmoid()
        
    def forward(self, G:DGLGraph):
        for layer in self.layers:
            G = layer(G)
        
        output = {}

        # distCB
        if self.distCB:
            output['distCB'] = self.output_layer(G.edata['efeat'])

        # global and local qa
        if self.QA:
            h_global = torch.cat([dgl.mean_nodes(G, 'nfeat'), dgl.mean_edges(G, 'efeat')], dim=1)
            y_global = self.sigmoid(self.global_qa_linear(h_global))
            output['global_lddt'] = y_global
            y_local = self.sigmoid(self.local_qa_linear(G.ndata['nfeat']))
            output['local_lddt'] = y_local
        
        return output