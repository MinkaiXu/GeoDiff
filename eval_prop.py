import os
import pickle
import argparse
import torch
import numpy as np
from psikit import Psikit
from tqdm.auto import tqdm
from easydict import EasyDict
from torch_geometric.data import Data

from utils.datasets import PackedConformationDataset
from utils.chem import set_rdmol_positions


class PropertyCalculator(object):

    def __init__(self, threads, memory, seed):
        super().__init__()
        self.pk = Psikit(threads=threads, memory=memory)
        self.seed = seed

    def __call__(self, data, num_confs=50):
        rdmol = data.rdmol
        confs = data.pos_prop

        conf_idx = np.arange(confs.shape[0])
        np.random.RandomState(self.seed).shuffle(conf_idx)
        conf_idx = conf_idx[:num_confs]

        data.prop_conf_idx = []
        data.prop_energy = []
        data.prop_homo = []
        data.prop_lumo = []
        data.prop_dipo = []

        for idx in tqdm(conf_idx):
            mol = set_rdmol_positions(rdmol, confs[idx])
            self.pk.mol = mol
            try:
                energy, homo, lumo, dipo = self.pk.energy(), self.pk.HOMO, self.pk.LUMO, self.pk.dipolemoment[-1]
                data.prop_conf_idx.append(idx)
                data.prop_energy.append(energy)
                data.prop_homo.append(homo)
                data.prop_lumo.append(lumo)
                data.prop_dipo.append(dipo)
            except:
                pass
        
        return data


def get_prop_matrix(data):
    """
    Returns:
        properties: (4, num_confs) numpy tensor. Energy, HOMO, LUMO, DipoleMoment
    """
    return np.array([
        data.prop_energy,
        data.prop_homo,
        data.prop_lumo,
        data.prop_dipo,
    ])


def get_ensemble_energy(props):
    """
    Args:
        props: (4, num_confs)
    """
    avg_ener = np.mean(props[0, :])
    low_ener = np.min(props[0, :])
    gaps = np.abs(props[1, :] - props[2, :])
    avg_gap = np.mean(gaps)
    min_gap = np.min(gaps)
    max_gap = np.max(gaps)
    return np.array([
        avg_ener, low_ener, avg_gap, min_gap, max_gap,
    ])

HART_TO_EV = 27.211
HART_TO_KCALPERMOL = 627.5 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./data/GEOM/QM9/qm9_property.pkl')
    parser.add_argument('--generated', type=str, default=None)
    parser.add_argument('--num_confs', type=int, default=50)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    prop_cal = PropertyCalculator(threads=args.threads, memory=args.memory, seed=args.seed)

    cache_ref_fn = os.path.join(
        os.path.dirname(args.dataset),
        os.path.basename(args.dataset)[:-4] + '_prop.pkl'
    )
    if not os.path.exists(cache_ref_fn):
        dset = PackedConformationDataset(args.dataset)
        dset = [data for data in dset]
        dset_prop = []
        for data in dset:
            data.pos_prop = data.pos_ref.reshape(-1, data.num_nodes, 3)
            dset_prop.append(prop_cal(data, args.num_confs))
        with open(cache_ref_fn, 'wb') as f:
            pickle.dump(dset_prop, f)
        dset = dset_prop
    else:
        with open(cache_ref_fn, 'rb') as f:
            dset = pickle.load(f)
    

    if args.generated is None:
        exit()

    print('Start evaluation.')

    cache_gen_fn = os.path.join(
        os.path.dirname(args.generated),
        os.path.basename(args.generated)[:-4] + '_prop.pkl'
    )
    if not os.path.exists(cache_gen_fn):
        with open(args.generated, 'rb') as f:
            gens = pickle.load(f)
        gens_prop = []
        for data in gens:
            if not isinstance(data, Data):
                data = EasyDict(data)
            data.num_nodes = data.rdmol.GetNumAtoms()
            data.pos_prop = data.pos_gen.reshape(-1, data.num_nodes, 3)
            gens_prop.append(prop_cal(data, args.num_confs))
        with open(cache_gen_fn, 'wb') as f:
            pickle.dump(gens_prop, f)
        gens = gens_prop
    else:
        with open(cache_gen_fn, 'rb') as f:
            gens = pickle.load(f)


    dset = {d.smiles:d for d in dset}
    gens = {d.smiles:d for d in gens}
    all_diff = []
    for smiles in dset.keys():
        if smiles not in gens:
            continue

        prop_gts = get_ensemble_energy(get_prop_matrix(dset[smiles])) * HART_TO_EV
        prop_gen = get_ensemble_energy(get_prop_matrix(gens[smiles])) * HART_TO_EV
        # prop_gts = np.mean(get_prop_matrix(dset[smiles]), axis=1)
        # prop_gen = np.mean(get_prop_matrix(gens[smiles]), axis=1)

        # print(get_prop_matrix(gens[smiles])[0])

        prop_diff = np.abs(prop_gts - prop_gen)

        print('\nProperty: %s' % smiles)
        print('  Gts :', prop_gts)
        print('  Gen :', prop_gen)
        print('  Diff:', prop_diff)
        
        all_diff.append(prop_diff.reshape(1, -1))
    all_diff = np.vstack(all_diff)  # (num_mols, 4)
    print(all_diff.shape)

    print('[Difference]')
    print('  Mean:  ', np.mean(all_diff, axis=0))
    print('  Median:', np.median(all_diff, axis=0))
    print('  Std:   ', np.std(all_diff, axis=0))
