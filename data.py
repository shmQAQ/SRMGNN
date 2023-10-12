import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from dgl.transforms import KHopGraph
from dgl.data.utils import save_graphs, load_graphs
from functools import partial
from dgllife.utils.io import pmap
from dgllife.utils.mol_to_graph import SMILESToBigraph


class MoleculeDataset(object):
    '''
    a class for loading molecular dataset from class pd.dataframe
    reading the data from csv file 
    molecules will be converted to dgl graph object
    and then also abstracted to k-hop subgraphs

    parameters:
    -----------
    df: pd.dataframe
        the dataframe containing the data
    smiles_to_graph: callable
        a function that converts smiles to dgl graph
    node_featurizer: callable
        a function that converts atom to node features
    edge_featurizer: callable
        a function that converts bond to edge features
    k: int
        the size of subgraphs extracted from molecules
    smiles_column: str
        the name of the column containing smiles
    cache_file_path: str
        the path to store the cache file
    task_names: list of str
        the names of the columns containing task values
    load: bool
        whether to load the cache file
    '''

    def __init__(self, df, smiles_to_graph=None, node_featurizer=None,
                 edge_featurizer=None, k=3, smiles_column='smiles',
                 cache_file_path=None, task_names=None, load=False):
        self.df = df
        self.smile_column = self.df[smiles_column].tolist()
        if self.task_names is None:
                self.task_names = self.df.columns.drop([self.smiles_column]).tolist()
        else:
                self.task_names = task_names
        self.num_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        
        if smiles_to_graph is None:
            self.smiles_to_graph = SMILESToBigraph(
                node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
        else:
            self.smiles_to_graph = partial(smiles_to_graph,
                                           node_featurizer=node_featurizer,
                                           edge_featurizer=edge_featurizer)
        self.process(self, smiles_to_graph, k, load)

    def k_hop_subgraph(self, g, k):
        '''
        extract k-hop subgraphs from a graph

        parameters:
        -----------
        g: DGLGraph
            the graph
        k: int
            the size of subgraphs

        returns:
        --------
        g_list: 
        the list of subgraphs
        '''
        g_list = []
        transform = KHopGraph(k)
        for node_id in range(g.number_of_nodes()):
            sg = transform(g, node_id)
            if sg.number_of_nodes() > 0:
                g_list.append(sg)
        return g_list
        
    def process(self, smiles_to_graph, k, load):
        '''
        process the data

        parameters:
        -----------
        smiles_to_graph: callable
            a function that converts smiles to dgl graph
        k: int
            the size of subgraphs
        load: bool
            whether to load the cache file
        '''
        if self.cache_file_path is not None and os.path.isfile(self.cache_file_path) and load:
            self.graphs, self.labels = load_graphs(self.cache_file_path)
            self.labels = self.labels[0].tolist()
        else:
            self.graphs = pmap(smiles_to_graph, self.smile_column)
            self.labels = self.df[self.task_names].values
            if self.cache_file_path is not None:
                save_graphs(self.cache_file_path, self.graphs, self.labels)
            self.sub_graphs = []
            for g in self.graphs:
                self.sub_graphs.extend(self.k_hop_subgraph(g, k))
            

    def __getitem__(self, idx):
        '''
        get the data

        parameters:
        -----------
        idx: int
            the index of data

        returns:
        --------
        g: DGLGraph
            the graph
        label: float
            the label
        '''
        return self.graphs[idx], self.sub_graphs[idx], self.labels[idx]
    
    def __len__(self):
        '''
        get the size of the dataset

        returns:
        --------
        len: int
            the size of the dataset
        '''
        return len(self.graphs)


