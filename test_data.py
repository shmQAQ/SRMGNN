import unittest
import os
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from data import MoleculeDataset

class TestMoleculeDataset(unittest.TestCase):
    def setUp(self):
        self.file_path = 'test_data.csv'
        with open(self.file_path, 'w') as f:
            f.write('smiles,task1,task2\n')
            f.write('CCO,1,0\n')
            f.write('CCN,0,1\n')
        self.dataset = MoleculeDataset(self.file_path, 'task1', node_featurizer=CanonicalAtomFeaturizer, 
                                       edge_featurizer=CanonicalBondFeaturizer, k=2, cache_path='test_cache')

    def tearDown(self):
        os.remove(self.file_path)
        for i in range(len(self.dataset)):
            filepath = "test_cache_{}.pt".format(i)
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        g = self.dataset[0]
        self.assertIsInstance(g, torch.Tensor)
        self.assertEqual(g.shape[0], 2)
        self.assertEqual(g.shape[1], 5)
        self.assertEqual(g.shape[2], 2)

    def test_generate_graphs(self):
        graphs = self.dataset.generate_graphs()
        self.assertEqual(len(graphs), 2)
        self.assertIsInstance(graphs[0], torch.Tensor)
        self.assertEqual(graphs[0].shape[0], 2)
        self.assertEqual(graphs[0].shape[1], 5)
        self.assertEqual(graphs[0].shape[2], 2)

    def test_generate_k_hop_graphs(self):
        k_hop_graphs = self.dataset.generate_k_hop_graphs(cache_path='test_cache')
        self.assertEqual(len(k_hop_graphs), 2)
        self.assertIsInstance(k_hop_graphs[0], torch.Tensor)
        self.assertEqual(k_hop_graphs[0].shape[0], 2)
        self.assertEqual(k_hop_graphs[0].shape[1], 5)
        self.assertEqual(k_hop_graphs[0].shape[2], 2)