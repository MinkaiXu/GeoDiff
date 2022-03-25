import os
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random

import torch
from torch_geometric.data import Data, Dataset, Batch

from torch_geometric.utils import to_networkx
from torch_scatter import scatter
#from torch.utils.data import Dataset

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
import networkx as nx
from tqdm import tqdm

# import sidechainnet as scn
RDLogger.DisableLog('rdApp.*')

from .chem import BOND_TYPES, mol_to_smiles

def prepare_pdb2(scn_dir, data_path):

    # step 1: filter and save pdb file.
    train_data = []
    cnt_fail = 0
    

    def get_num_plusseg(msk):
        tmp = [0]
        for i in range(1, len(msk)):
            if msk[i] == msk[i-1]:
                tmp.append(0)
            else:
                tmp.append(1)
        s = sum(tmp)
        if msk[0] == '-':
            return (s + 1) // 2
        else:
            return (s // 2) + 1        
        
    def get_plus_rate(msk):
        cnt = sum([1 if x == '+' else 0 for x in msk])
        return cnt / len(msk)
    
    d = scn.load(casp_version=12, thinning=30, scn_dir=scn_dir)
    raw_data = d['train']

    mask = raw_data['msk']
    n_raw_data = len(mask)
    cnt_seg = 0
    cnt_success = 0
    for i in tqdm(range(n_raw_data)):
        if get_plus_rate(mask[i]) > 0.5 and get_num_plusseg(mask[i]) == 1:
            cnt_seg += 1
            mask_ = [1 if _ == '+' else 0 for _ in mask[i]]
            if sum(mask_) < 200:
                cnt_success += 1                
                seq = raw_data['seq'][i]
                crd = raw_data['crd'][i]
                name = raw_data['ids'][i]
                mol = scn.StructureBuilder(seq, crd)
                mol.to_pdb('./tmp.pdb')
                data = pdb_to_data('./tmp.pdb', name)
                if data is not None:
                    train_data.append(data)
                else:
                    cnt_fail += 1                
    
    print('total n_raw_data: %d, cnt_seg: %d, cnt_success: %d' % (n_raw_data, cnt_seg, cnt_success))
    
    n_data = len(train_data)
    print('number of train samples: %d | number of fails: %d' % (n_data, cnt_fail))

    os.makedirs(os.path.join(data_path), exist_ok=True)

    with open(os.path.join(data_path, 'train_data_%dk.pkl' % (n_data // 1000)), "wb") as fout:
        pickle.dump(train_data, fout)
    print('save train %dk done' % (n_data // 1000))

 

def prepare_pdblarge(scn_dir, data_path):

    # step 1: filter and save pdb file.
    train_data = []
    cnt_fail = 0
    
    max_residue = 0
    
    d = scn.load(casp_version=12, thinning=30, scn_dir=scn_dir)
    raw_data = d['train']

    mask = raw_data['msk']
    n_raw_data = len(mask)
    cnt_seg = 0
    cnt_success = 0
    for i in tqdm(range(n_raw_data)):
        # if get_plus_rate(mask[i]) > 0.5 and get_num_plusseg(mask[i]) == 1:
        if True:
            cnt_seg += 1
            mask_ = [1 if _ == '+' else 0 for _ in mask[i]]
            if sum(mask_) < 400:
                
                cnt_success += 1                
                seq = raw_data['seq'][i]
                crd = raw_data['crd'][i]
                name = raw_data['ids'][i]
                mol = scn.StructureBuilder(seq, crd)
                mol.to_pdb('./tmp.pdb')
                data = pdb_to_data('./tmp.pdb', name)
                if data is not None:
                    train_data.append(data)
                    max_residue = max(max_residue, sum(mask_))
                else:
                    cnt_fail += 1                
    
    print('total n_raw_data: %d, cnt_seg: %d, cnt_success: %d, max_residue: %d' % (n_raw_data, cnt_seg, cnt_success, max_residue))
    
    n_data = len(train_data)
    print('number of train samples: %d | number of fails: %d' % (n_data, cnt_fail))

    os.makedirs(os.path.join(data_path), exist_ok=True)

    with open(os.path.join(data_path, 'train_data_%dk.pkl' % (n_data // 1000)), "wb") as fout:
        pickle.dump(train_data, fout)
    print('save train %dk done' % (n_data // 1000))

 
def prepare_pdb_valtest(scn_dir, data_path):

    # step 1: filter and save pdb file.
    val_data = []
    test_data = []
    all_data = []

    cnt_fail = 0
    
    max_residue = 0
    n_raw_data = 0
    cnt_success = 0    
    
    d = scn.load(casp_version=12, thinning=30, scn_dir=scn_dir)
    fetch_dict = ['test', 'valid-10', 'valid-20', 'valid-30', 'valid-40', 'valid-50', 'valid-70', 'valid-90']
    for dict_name in fetch_dict:
        raw_data = d[dict_name]
        mask = raw_data['msk']
        n_raw_data += len(mask)
        cnt_seg = 0
        cnt_success = 0
        for i in tqdm(range(len(mask))):
            # if get_plus_rate(mask[i]) > 0.5 and get_num_plusseg(mask[i]) == 1:
            if True:
                mask_ = [1 if _ == '+' else 0 for _ in mask[i]]
                if sum(mask_) < 400:
                    
                    seq = raw_data['seq'][i]
                    crd = raw_data['crd'][i]
                    name = raw_data['ids'][i]
                    mol = scn.StructureBuilder(seq, crd)
                    mol.to_pdb('./tmp.pdb')
                    data = pdb_to_data('./tmp.pdb', name)
                    if data is not None:
                        cnt_success += 1                
                        all_data.append(data)
                        max_residue = max(max_residue, sum(mask_))
                    else:
                        cnt_fail += 1                
    
    print('total n_raw_data: %d, cnt_success: %d, max_residue: %d' % (n_raw_data, cnt_success, max_residue))
    
    random.shuffle(all_data)
    n_val = len(all_data) // 2
    n_test = len(all_data) - n_val
    print('number of val samples: %d | number of test samples: %d | number of fails: %d' % (n_val, n_test, cnt_fail))

    os.makedirs(os.path.join(data_path), exist_ok=True)

    with open(os.path.join(data_path, 'val_data_%dk.pkl' % (n_val // 1000)), "wb") as fout:
        pickle.dump(all_data[:n_val], fout)
    print('save val %dk done' % (n_val // 1000))

    with open(os.path.join(data_path, 'test_data_%dk.pkl' % (n_test // 1000)), "wb") as fout:
        pickle.dump(all_data[n_val:], fout)
    print('save test %dk done' % (n_test // 1000))





def pdb_to_data(pdb_path, name):
    mol = Chem.rdmolfiles.MolFromPDBFile(pdb_path)
    if mol is None:
        return None
    with open(pdb_path, 'r') as f:
        pdb_infos = f.readlines()    
    pdb_infos = pdb_infos[1:-1]        

    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()
    # name = pdb_path.split('/')[-1].split('.')[0]
    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    is_sidechain = []
    is_alpha = []
    atom2res = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    
    for index, atom in enumerate(mol.GetAtoms()):
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

        info = atom.GetPDBResidueInfo()
        ref_info = pdb_infos[index]
        ref_info = ref_info.split()

        assert info.GetResidueName().strip() == ref_info[3]
        assert info.GetName().strip() == ref_info[2]
        assert info.GetResidueNumber() == int(ref_info[4])
        if info.GetName().strip() == 'CA':
            is_alpha.append(1)
        else:
            is_alpha.append(0)

        if info.GetName().strip() in ['N', 'CA', 'C', 'O']:
            is_sidechain.append(0)
        else:
            is_sidechain.append(1)
        atom2res.append(info.GetResidueNumber() - 1)
    
    num_res = len(set(atom2res))
    atom2res = np.array(atom2res)
    atom2res -= atom2res.min()
    atom2res = torch.tensor(atom2res, dtype=torch.long)
    is_sidechain = torch.tensor(is_sidechain).bool()
    is_alpha = torch.tensor(is_alpha).bool()

    dummy_index = torch.arange(pos.size(0))
    alpha_index = dummy_index[is_alpha]
    res2alpha_index = -torch.ones(5000, dtype=torch.long)
    res2alpha_index[atom2res[is_alpha]] = alpha_index
    atom2alpha_index = res2alpha_index[atom2res]

    if is_sidechain.sum().item() == 0: # protein built solely on GLY can not be used for sidechain prediction
        return None
    # assert (4 * num_res == (len(is_sidechain) - sum(is_sidechain))),(4 * num_res, (len(is_sidechain) - sum(is_sidechain)))
    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)
    if edge_index.size(1) == 0: # only alpha carbon
        return None
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    # smiles = Chem.MolToSmiles(mol)

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type, is_alpha=is_alpha,
                rdmol=copy.deepcopy(mol), name=name, is_sidechain=is_sidechain, atom2res=atom2res, atom2alpha_index=atom2alpha_index)
    #data.nx = to_networkx(data, to_undirected=True)

    return data


def rdmol_to_data(mol:Mol, smiles=None, data_cls=Data):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    data = data_cls(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    #data.nx = to_networkx(data, to_undirected=True)

    return data


class MolClusterData(Data):

    def __inc__(self, key, value):
        if key == 'subgraph_index':
            return self.subgraph_index.max().item() + 1
        else:
            return super().__inc__(key, value)


def rdmol_cluster_to_data(mol:Mol, smiles=None):
    data = rdmol_to_data(mol, smiles, data_cls=MolClusterData)
    data.subgraph_index = torch.zeros([data.atom_type.size(0)], dtype=torch.long)    
    for i, subgraph in enumerate(nx.connected_components(to_networkx(data, to_undirected=True))):
        data.subgraph_index[list(subgraph)] = i
    return data


def preprocess_iso17_dataset(base_path):
    train_path = os.path.join(base_path, 'iso17_split-0_train.pkl')
    test_path = os.path.join(base_path, 'iso17_split-0_test.pkl')
    with open(train_path, 'rb') as fin:
        raw_train = pickle.load(fin)
    with open(test_path, 'rb') as fin:
        raw_test = pickle.load(fin)

    smiles_list_train = [mol_to_smiles(mol) for mol in raw_train]
    smiles_set_train = list(set(smiles_list_train))
    smiles_list_test = [mol_to_smiles(mol) for mol in raw_test]
    smiles_set_test = list(set(smiles_list_test))

    print('preprocess train...')
    all_train = []
    for i in tqdm(range(len(raw_train))):
        smiles = smiles_list_train[i]
        data = rdmol_to_data(raw_train[i], smiles=smiles)
        all_train.append(data)

    print('Train | find %d molecules with %d confs' % (len(smiles_set_train), len(all_train)))    
    
    print('preprocess test...')
    all_test = []
    for i in tqdm(range(len(raw_test))):
        smiles = smiles_list_test[i]
        data = rdmol_to_data(raw_test[i], smiles=smiles)
        all_test.append(data)

    print('Test | find %d molecules with %d confs' % (len(smiles_set_test), len(all_test)))  

    return all_train, all_test



def preprocess_GEOM_dataset(base_path, dataset_name, max_conf=5, train_size=0.8, max_size=9999999999, seed=None):

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        num_mols += 1
        num_confs += min(max_conf, u_conf)
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)
        if num_mols >= max_size:
            break
    print('pre-filter: find %d molecules with %d confs' % (num_mols, num_confs))

    # 1. select maximal 'max_conf' confs of each qm9 molecule
    # 2. split the dataset based on 2d-structure, i.e., test on unseen graphs
    train_data, val_data, test_data = [], [], []
    val_size = test_size = (1. - train_size) / 2

    num_mols = np.zeros(4, dtype=int) # (tot, train, val, test)
    num_confs = np.zeros(4, dtype=int) # (tot, train, val, test)

    '''
    # mol.get('uniqueconfs') != len(mol.get('conformers'))
    with open(os.path.join(base_path, pickle_path_list[1878]), 'rb') as fin:
        mol = pickle.load(fin)
    print(mol.get('uniqueconfs'), len(mol.get('conformers')))
    print(mol.get('conformers')[0]['rd_mol'].GetConformer(0).GetPositions())
    print(mol.get('conformers')[1]['rd_mol'].GetConformer(0).GetPositions())
    return 
    '''

    bad_case = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')


        if mol.get('uniqueconfs') <= max_conf:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'max_conf' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:max_conf]

        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'))
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor([v], dtype=torch.float32)
            datas.append(data)

        # split
        eps = np.random.rand()
        if eps <= train_size:
            train_data.extend(datas)
            num_mols += [1, 1, 0, 0]
            num_confs += [len(datas), len(datas), 0, 0]
        elif eps <= train_size + val_size:
            val_data.extend(datas)
            num_mols += [1, 0, 1, 0]
            num_confs += [len(datas), 0, len(datas), 0]
        else:
            test_data.extend(datas)
            num_mols += [1, 0, 0, 1]
            num_confs += [len(datas), 0, 0, len(datas)]

    print('post-filter: find %d molecules with %d confs' % (num_mols[0], num_confs[0]))    
    print('train size: %d molecules with %d confs' % (num_mols[1], num_confs[1]))    
    print('val size: %d molecules with %d confs' % (num_mols[2], num_confs[2]))    
    print('test size: %d molecules with %d confs' % (num_mols[3], num_confs[3]))    
    print('bad case: %d' % bad_case)
    print('done!')

    return train_data, val_data, test_data
    




def preprocess_GEOM_dataset_with_fixed_num_conf(base_path, dataset_name, conf_per_mol=5, train_size=0.8, tot_mol_size=50000, seed=None):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name, should be in [qm9, drugs]
    conf_per_mol: keep mol that has at least conf_per_mol confs, and sampling the most probable conf_per_mol confs
    train_size ratio, val = test = (1-train_size) / 2
    tot_mol_size: max num of mols. The total number of final confs should be tot_mol_size * conf_per_mol
    seed: rand seed for RNG
    """

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < conf_per_mol:
            continue
        num_mols += 1
        num_confs += conf_per_mol
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)
        # we need do a shuffle and sample first max_size items here.
        #if num_mols >= max_size:
        #    break
    random.shuffle(pickle_path_list)
    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)

    pickle_path_list = pickle_path_list[:tot_mol_size]

    print('pre-filter: find %d molecules with %d confs, use %d molecules with %d confs' % (num_mols, num_confs, tot_mol_size, tot_mol_size*conf_per_mol))


    # 1. select maximal 'max_conf' confs of each qm9 molecule
    # 2. split the dataset based on 2d-structure, i.e., test on unseen graphs
    train_data, val_data, test_data = [], [], []
    val_size = test_size = (1. - train_size) / 2

    # generate train, val, test split indexes
    split_indexes = list(range(tot_mol_size))
    random.shuffle(split_indexes)
    index2split = {}
    #print(int(len(split_indexes) * train_size), int(len(split_indexes) * (train_size + val_size)), len(split_indexes))
    for i in range(0, int(len(split_indexes) * train_size)):
        index2split[split_indexes[i]] = 'train'
    for i in range(int(len(split_indexes) * train_size), int(len(split_indexes) * (train_size + val_size))):
        index2split[split_indexes[i]] = 'val'
    for i in range(int(len(split_indexes) * (train_size + val_size)), len(split_indexes)):
        index2split[split_indexes[i]] = 'test'        


    num_mols = np.zeros(4, dtype=int) # (tot, train, val, test)
    num_confs = np.zeros(4, dtype=int) # (tot, train, val, test)


    bad_case = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        if mol.get('uniqueconfs') == conf_per_mol:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'max_conf' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:conf_per_mol]

        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor([v], dtype=torch.float32)
            data['idx'] = torch.tensor([i], dtype=torch.long)
            datas.append(data)
        assert len(datas) == conf_per_mol

        # split
        '''
        eps = np.random.rand()
        if eps <= train_size:
            train_data.extend(datas)
            num_mols += [1, 1, 0, 0]
            num_confs += [len(datas), len(datas), 0, 0]
        elif eps <= train_size + val_size:
            val_data.extend(datas)
            num_mols += [1, 0, 1, 0]
            num_confs += [len(datas), 0, len(datas), 0]
        else:
            test_data.extend(datas)
            num_mols += [1, 0, 0, 1]
            num_confs += [len(datas), 0, 0, len(datas)]
        '''

        if index2split[i] == 'train':
            train_data.extend(datas)
            num_mols += [1, 1, 0, 0]
            num_confs += [len(datas), len(datas), 0, 0]
        elif index2split[i] == 'val':    
            val_data.extend(datas)
            num_mols += [1, 0, 1, 0]
            num_confs += [len(datas), 0, len(datas), 0]
        elif index2split[i] == 'test': 
            test_data.extend(datas)
            num_mols += [1, 0, 0, 1]
            num_confs += [len(datas), 0, 0, len(datas)] 
        else:
            raise ValueError('unknown index2split value.')                         

    print('post-filter: find %d molecules with %d confs' % (num_mols[0], num_confs[0]))    
    print('train size: %d molecules with %d confs' % (num_mols[1], num_confs[1]))    
    print('val size: %d molecules with %d confs' % (num_mols[2], num_confs[2]))    
    print('test size: %d molecules with %d confs' % (num_mols[3], num_confs[3]))    
    print('bad case: %d' % bad_case)
    print('done!')

    return train_data, val_data, test_data, index2split


def get_test_set_with_large_num_conf(base_path, dataset_name, block, tot_mol_size=1000, seed=None, confmin=50, confmax=500):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name, should be in [qm9, drugs]
    conf_per_mol: keep mol that has at least conf_per_mol confs, and sampling the most probable conf_per_mol confs
    train_size ratio, val = test = (1-train_size) / 2
    tot_mol_size: max num of mols. The total number of final confs should be tot_mol_size * conf_per_mol
    seed: rand seed for RNG
    """
    #block smiles in train / val 
    block_smiles = defaultdict(int)
    for i in range(len(block)):
        block_smiles[block[i].smiles] = 1

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < confmin or u_conf > confmax:
            continue
        if block_smiles[smiles] == 1:
            continue

        num_mols += 1
        num_confs += u_conf
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)
        # we need do a shuffle and sample first max_size items here.
        #if num_mols >= tot_mol_size:
        #    break

    random.shuffle(pickle_path_list)
    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)

    pickle_path_list = pickle_path_list[:tot_mol_size]

    print('pre-filter: find %d molecules with %d confs' % (num_mols, num_confs))


  

    bad_case = 0
    all_test_data = []
    num_valid_mol = 0
    num_valid_conf = 0

    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        conf_ids = np.arange(mol.get('uniqueconfs'))
      
        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor([v], dtype=torch.float32)
            data['idx'] = torch.tensor([i], dtype=torch.long)
            datas.append(data)

      
        all_test_data.extend(datas)
        num_valid_mol += 1
        num_valid_conf += len(datas)

    print('poster-filter: find %d molecules with %d confs' % (num_valid_mol, num_valid_conf))


    return all_test_data


class ConformationDataset(Dataset):

    def __init__(self, path, transform=None):
        super().__init__()
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):

        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.data)

        
    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)

class SidechainConformationDataset(ConformationDataset):

    def __init__(self, path, transform=None, cutoff=10., max_residue=5000, fix_subgraph=False):
        super().__init__(path, transform)
        self.cutoff = cutoff
        self.max_residue = max_residue
        self.fix_subgraph = fix_subgraph


    def __getitem__(self, idx):

        data = self.data[idx].clone()
        """ Subgraph sampling
            1. sampling an atom from the backbone (residue)
            2. Find all neighboring atoms within a cutoff
            3. extend atoms to ensure the completeness of each residue
            4. remap the index for subgraph
        """
        is_sidechain = data.is_sidechain
        pos = data.pos
        edge_index = data.edge_index
        atom2res = data.atom2res
        dummy_index = torch.arange(pos.size(0))
        backbone_index = dummy_index[~is_sidechain]


        #stop=False
        #while not stop:
        # step 1
        if self.fix_subgraph:
            center_atom_index = backbone_index[backbone_index.size(0) // 2].view(1,)
        else:
            center_atom_index = backbone_index[torch.randint(low=0, high=backbone_index.size(0), size=(1, ))] # (1, )
        pos_center_atom = pos[center_atom_index] # (1, 3)
        # step 2
        distance = (pos_center_atom - pos).norm(dim=-1)
        mask = (distance <= self.cutoff)
        # step 3
        is_keep_residue = scatter(mask, atom2res, dim=-1, dim_size=self.max_residue, reduce='sum') # (max_residue, )
        is_keep_atom = is_keep_residue[atom2res]
        is_keep_edge = (is_keep_atom[edge_index[0]]) & (is_keep_atom[edge_index[1]])
        # step 4
        mapping = -torch.ones(pos.size(0), dtype=torch.long)
        keep_index = dummy_index[is_keep_atom]
        mapping[keep_index] = torch.arange(keep_index.size(0))
        if (data.is_sidechain[is_keep_atom]).sum().item() == 0:
            #stop = True
            return None

        # return subgraph data
        subgraph_data = Data(atom_type=data.atom_type[is_keep_atom], 
                             pos=data.pos[is_keep_atom], 
                             edge_index=mapping[data.edge_index[:, is_keep_edge]], 
                             edge_type=data.edge_type[is_keep_edge],
                             is_sidechain=data.is_sidechain[is_keep_atom], 
                             atom2res=data.atom2res[is_keep_atom])        

        if self.transform is not None:
            subgraph_data = self.transform(subgraph_data)        
        return subgraph_data

    @staticmethod
    def collate_fn(data):

        batch = [_ for _ in data if _ is not None]
        return Batch.from_data_list(batch)        



   


def accumulate_grad_from_subgraph(model, atom_type, pos, bond_index, bond_type, batch, atom2res, batch_size=8, device='cuda:0',
                                  is_sidechain=None, is_alpha=None, pos_gt=None, cutoff=10., max_residue=5000, transform=None):
    """
    1. decompose the protein to subgraphs
    2. evaluate subgraphs using trained models
    3. accumulate atom-wise grads
    4. return grads
    """

    accumulated_grad = torch.zeros_like(pos)
    accumulated_time = torch.zeros(pos.size(0), device=pos.deivce)

    all_subgraphs = []
    dummy_index = torch.arange(pos.size(0))
    
    # prepare subgraphs
    is_covered = torch.zeros(pos.size(0), device=pos.deivce).bool()
    is_alpha_and_uncovered = is_alpha & (~is_covered)
    while is_alpha_and_uncovered.sum().item() != 0:

        alpha_index = dummy_index[is_alpha_and_uncovered]
        center_atom_index = alpha_index[torch.randint(low=0, high=alpha_index.size(0), size=(1, ))] # (1, )
        pos_center_atom = pos[center_atom_index] # (1, 3)

        distance = (pos_center_atom - pos).norm(dim=-1)
        mask = (distance <= cutoff)

        is_keep_residue = scatter(mask, atom2res, dim=-1, dim_size=max_residue, reduce='sum') # (max_residue, )
        is_keep_atom = is_keep_residue[atom2res]
        is_keep_edge = (is_keep_atom[bond_index[0]]) & (is_keep_atom[bond_index[1]])

        mapping = -torch.ones(pos.size(0), dtype=torch.long)
        keep_index = dummy_index[is_keep_atom]
        mapping[keep_index] = torch.arange(keep_index.size(0))
    
        is_covered |= is_keep_atom
        is_alpha_and_uncovered = is_alpha & (~is_covered)   

        if (is_sidechain[is_keep_atom]).sum().item() == 0:
            continue

        subgraph = Data(atom_type=atom_type[is_keep_atom], 
                             pos=pos[is_keep_atom], 
                             edge_index=mapping[bond_index[:, is_keep_edge]], 
                             edge_type=bond_type[is_keep_edge],
                             is_sidechain=is_sidechain[is_keep_atom], 
                             atom2res=atom2res[is_keep_atom],
                             mapping=keep_index)    
        if transform is not None:
            subgraph = transform(subgraph)          
        all_subgraphs.append(subgraph)
    
    # run model
    tot_iters = (len(all_subgraphs) + batch_size - 1) // batch_size
    for it in range(tot_iters):
        batch = Batch.from_data_list(all_subgraphs[it * batch_size, (it + 1) * batch_size]).to(device)
















class PackedConformationDataset(ConformationDataset):

    def __init__(self, path, transform=None):
        super().__init__(path, transform)
        #k:v = idx: data_obj
        self._pack_data_by_mol()

    def _pack_data_by_mol(self):
        """
        pack confs with same mol into a single data object
        """
        self._packed_data = defaultdict(list)
        if hasattr(self.data, 'idx'):
            for i in range(len(self.data)):
                self._packed_data[self.data[i].idx.item()].append(self.data[i])
        else:
            for i in range(len(self.data)):
                self._packed_data[self.data[i].smiles].append(self.data[i])
        print('[Packed] %d Molecules, %d Conformations.' % (len(self._packed_data), len(self.data)))

        new_data = []
        # logic
        # save graph structure for each mol once, but store all confs 
        cnt = 0
        for k, v in self._packed_data.items():
            data = copy.deepcopy(v[0])
            all_pos = []
            for i in range(len(v)):
                all_pos.append(v[i].pos)
            data.pos_ref = torch.cat(all_pos, 0) # (num_conf*num_node, 3)
            data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
            #del data.pos

            if hasattr(data, 'totalenergy'):
                del data.totalenergy
            if hasattr(data, 'boltzmannweight'):
                del data.boltzmannweight
            new_data.append(data)
        self.new_data = new_data

        

    def __getitem__(self, idx):

        data = self.new_data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.new_data)

