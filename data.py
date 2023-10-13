import os
import pandas as pd
import torch
import torch_geometric.utils as utils
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph

class MoleculeDataset(object):
    '''
    define the dataset for molecule
    input: 
        file_path: the path of the dataset
        task_name: the name of the task
        node_featurizer: the featurizer for node: dgllife.utils.CanonicalAtomFeaturizer
        edge_featurizer: the featurizer for edge: dgllife.utils.CanonicalBondFeaturizer
        k: the hop number for the k_hop graph
           
    process: 
        read the dataset from the file_path
        extract smiles and labels
        generate the graph data
        generate the k_hop graph data
    output: 
        the graph data
    '''

    def __init__(self, file_path: str, task_name: str, node_featurizer=CanonicalAtomFeaturizer, 
                 edge_featurizer=CanonicalBondFeaturizer, k: int = 3, cache_path=None, use_subgraph_edge_attr=True):
        self.file_path = file_path
        self.task_name = task_name
        self.cache_path = cache_path
        self.pd_data = pd.read_csv(self.file_path)
        self.smiles = self.pd_data['smiles']
        self.labels = self.pd_data[self.task_name]
        self.node_featurizer = node_featurizer()
        self.edge_featurizer = edge_featurizer()
        self.k = k
        self.graphs = self.generate_graphs()
        self.k_hop_graphs = self.generate_k_hop_graphs(cache_path, use_subgraph_edge_attr)
        self.degree = self.get_degree()

    def __len__(self):
        return len(self.smiles)
    
    def __inc__(self, key, value):
        if key == 'subgraph_edge_index':
            return self.num_subgraph_nodes 
        if key == 'subgraph_node_idx':
            return self.num_nodes 
        if key == 'subgraph_indicator':
            return self.num_nodes 
        elif 'index' in key:
            return self.num_nodes
        else:
            return 0
    
    def __getitem__(self, idx):
        return self.graphs[idx]

    def generate_graphs(self):
        graphs = []
        for i in range(len(self.smiles)):
            g = smiles_to_bigraph(self.smiles[i], node_featurizer=self.node_featurizer, 
                                  edge_featurizer=self.edge_featurizer)
            g = self.add_label(g, self.labels[i])
            g = g.from_dgl(g)
            graphs.append(g)
        return graphs
    
    def generate_k_hop_graphs(self, cache_path=None, use_subgraph_edge_attr=True):
        print("Extracting {}-hop subgraphs...".format(self.k))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        self.subgraph_node_index = []

        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
        self.subgraph_edge_index = []

        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
        self.subgraph_indicator_index = []

        # This gets the edge attributes for the new indices
        if use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(len(self)):
            if cache_path is not None:
                filepath = "{}_{}.pt".format(cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            edge_index_start = 0

            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx, 
                    self.k, 
                    graph.edge_index,
                    relabel_nodes=True, 
                    num_nodes=graph.num_nodes
                    )
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask]) # CHECK THIS DIDN"T BREAK ANYTHING
                edge_index_start += len(sub_nodes)

            if cache_path is not None:
                if use_subgraph_edge_attr and graph.edge_attr is not None:
                    subgraph_edge_attr = torch.cat(edge_attributes)
                else:
                    subgraph_edge_attr = None
                torch.save({
                    'subgraph_node_index': torch.cat(node_indices),
                    'subgraph_edge_index': torch.cat(edge_indices, dim=1),
                    'subgraph_indicator_index': torch.cat(indicators).type(torch.LongTensor),
                    'subgraph_edge_attr': subgraph_edge_attr
                }, filepath)
            else:
                self.subgraph_node_index.append(torch.cat(node_indices))
                self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
                self.subgraph_indicator_index.append(torch.cat(indicators))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("Done!")

    def get_degree(self):
        if not self.degree:
            self.degree_list = None
            return
        self.degree_list = []
        for g in self.graphs:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)