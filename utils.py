# _*_coding:utf-8_*_
import torch
import torch_geometric
from rdkit import Chem, Rdlogger
from torch_geometric.data import Data


def smile_to_graph(smiles: str, use_hydrogen: bool = False) -> torch_geometric.data.Data:
    '''
    convert smiles to graph
    :param smiles: smiles string
    :param use_hydrogen: whether to use hydrogen
    :return: graph
    '''
    x_map = {
        'atomic_num':
        list(range(0, 119)),
        'chirality': [
            'CHI_UNSPECIFIED',
            'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW',
            'CHI_OTHER',
            'CHI_TETRAHEDRAL',
            'CHI_ALLENE',
            'CHI_SQUAREPLANAR',
            'CHI_TRIGONALBIPYRAMIDAL',
            'CHI_OCTAHEDRAL',
        ],
        'degree':
        list(range(0, 11)),
        'formal_charge':
        list(range(-5, 7)),
        'num_hs':
        list(range(0, 9)),
        'num_radical_electrons':
        list(range(0, 5)),
        'hybridization': [
            'UNSPECIFIED',
            'S',
            'SP',
            'SP2',
            'SP3',
            'SP3D',
            'SP3D2',
            'OTHER',
        ],
        'is_aromatic': [False, True],
        'is_in_ring': [False, True],
    }


    e_map = {
        'bond_type': [
            'UNSPECIFIED',
            'SINGLE',
            'DOUBLE',
            'TRIPLE',
            'QUADRUPLE',
            'QUINTUPLE',
            'HEXTUPLE',
            'ONEANDAHALF',
            'TWOANDAHALF',
            'THREEANDAHALF',
            'FOURANDAHALF',
            'FIVEANDAHALF',
            'AROMATIC',
            'IONIC',
            'HYDROGEN',
            'THREECENTER',
            'DATIVEONE',
            'DATIVE',
            'DATIVEL',
            'DATIVER',
            'OTHER',
            'ZERO',
        ],
        'stereo': [
            'STEREONONE',
            'STEREOANY',
            'STEREOZ',
            'STEREOE',
            'STEREOCIS',
            'STEREOTRANS',
        ],
        'is_conjugated': [False, True],
    }
    Rdlogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Could not parse smiles string:', smiles)
    if use_hydrogen:
        mol = Chem.AddHs(mol)
    xs = []
    for atom in mol.GetAtoms():
            x = []
            x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
            x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
            x.append(x_map['degree'].index(atom.GetTotalDegree()))
            x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
            x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
            x.append(x_map['num_radical_electrons'].index(
                atom.GetNumRadicalElectrons()))
            x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
            x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
            x.append(x_map['is_in_ring'].index(atom.IsInRing()))
            xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
