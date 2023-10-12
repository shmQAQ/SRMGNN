import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data


def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    binary_encoding = [int(x == element) for element in allowable_set]
    return binary_encoding


def atom_featurer(atom,  use_chirality = True,
                  add_Hs = False):
    '''
    input:rdkit.Chem.rdchem.Atom
    output:atom feature np.array
    
    '''

    #define period table
    permitted_atoms = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'As', 'I', 'B',  'Sb', 'Sn', 
        'Se', 'H', 'Unknown'
    ]
    #compute atom feature
    atom_type = one_hot_encoding(str(atom.GetSymbol()), permitted_atoms)
    atom_degree = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, 5, 6, 7 ,8])
    atom_formal_charge = one_hot_encoding(int(atom.GetFormalCharge()), [-1, -2, -3 , -4, 1, 2, 3, 4 ,0,'More than 4'])
    hybridization_type = one_hot_encoding(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'S', 'UNSPECIFIED'])
    is_in_ring = [int(atom.IsInRing())]
    is_aromatic = [int(atom.GetIsAromatic())]
    atomic_mass = [float(atom.GetMass())]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

    atom_feature_vector = atom_type + atom_degree + atom_formal_charge + hybridization_type + is_in_ring + is_aromatic + atomic_mass + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality = one_hot_encoding(str(atom.GetChiralTag()), ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'])
        atom_feature_vector = atom_feature_vector + chirality

    if add_Hs == True:
        num_of_Hs = [int(atom.GetTotalNumHs())]
        atom_feature_vector = atom_feature_vector + num_of_Hs

    return np.array(atom_feature_vector)


def bond_featurer(bond,
                  use_stereo = True):
    '''
    input:rdkit.Chem.rdchem.Bond
    output:bond feature np.array
    '''
    bond_type = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_encoding = one_hot_encoding(bond.GetBondType(), bond_type)
    bond_conj = [int(bond.GetIsConjugated())]
    bond_ring = [int(bond.IsInRing())]
    bond_feature = bond_type_encoding + bond_conj + bond_ring

    if use_stereo == True:
        bond_stereo = one_hot_encoding(str(bond.GetStereo()), ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'])
        bond_feature = bond_feature + bond_stereo
    
    return np.array(bond_feature)

def Smiles2PygData(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(atom_featurer(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(bond_featurer(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = atom_featurer(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = bond_featurer(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))

    return data_list