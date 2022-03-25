import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from torch_geometric.data import Data
from functools import partial 
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from ..chem import set_rdmol_positions, get_best_rmsd


def get_rmsd_confusion_matrix(data: Data, useFF=False):
    data['pos_ref'] = data['pos_ref'].reshape(-1, data['rdmol'].GetNumAtoms(), 3)
    data['pos_gen'] = data['pos_gen'].reshape(-1, data['rdmol'].GetNumAtoms(), 3)
    num_gen = data['pos_gen'].shape[0]
    num_ref = data['pos_ref'].shape[0]

    # assert num_gen == data.num_pos_gen.item()
    # assert num_ref == data.num_pos_ref.item()

    rmsd_confusion_mat = -1 * np.ones([num_ref, num_gen],dtype=np.float)
    
    for i in range(num_gen):
        gen_mol = set_rdmol_positions(data['rdmol'], data['pos_gen'][i])
        if useFF:
            #print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(gen_mol)
        for j in range(num_ref):
            ref_mol = set_rdmol_positions(data['rdmol'], data['pos_ref'][j])
            
            rmsd_confusion_mat[j,i] = get_best_rmsd(gen_mol, ref_mol)

    return rmsd_confusion_mat
    

def evaluate_conf(data: Data, useFF=False, threshold=0.5):
    rmsd_confusion_mat = get_rmsd_confusion_matrix(data, useFF=useFF)
    rmsd_ref_min = rmsd_confusion_mat.min(-1)
    #print('done one mol')
    #print(rmsd_ref_min)
    return (rmsd_ref_min<=threshold).mean(), rmsd_ref_min.mean()


def print_covmat_results(results, print_fn=print):
    df = pd.DataFrame({
        'COV-R_mean': np.mean(results.CoverageR, 0),
        'COV-R_median': np.median(results.CoverageR, 0),
        'COV-R_std': np.std(results.CoverageR, 0),
        'COV-P_mean': np.mean(results.CoverageP, 0),
        'COV-P_median': np.median(results.CoverageP, 0),
        'COV-P_std': np.std(results.CoverageP, 0),
    }, index=results.thresholds)
    print_fn('\n' + str(df))
    print_fn('MAT-R_mean: %.4f | MAT-R_median: %.4f | MAT-R_std %.4f' % (
        np.mean(results.MatchingR), np.median(results.MatchingR), np.std(results.MatchingR)
    ))
    print_fn('MAT-P_mean: %.4f | MAT-P_median: %.4f | MAT-P_std %.4f' % (
        np.mean(results.MatchingP), np.median(results.MatchingP), np.std(results.MatchingP)
    ))
    return df



class CovMatEvaluator(object):

    def __init__(self, 
        num_workers=8, 
        use_force_field=False, 
        thresholds=np.arange(0.05, 3.05, 0.05),
        ratio=2,
        filter_disconnected=True,
        print_fn=print,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.use_force_field = use_force_field
        self.thresholds = np.array(thresholds).flatten()
        
        self.ratio = ratio
        self.filter_disconnected = filter_disconnected
        
        self.pool = mp.Pool(num_workers)
        self.print_fn = print_fn

    def __call__(self, packed_data_list, start_idx=0):
        func = partial(get_rmsd_confusion_matrix, useFF=self.use_force_field)
        
        filtered_data_list = []
        for data in packed_data_list:
            if 'pos_gen' not in data or 'pos_ref' not in data: continue
            if self.filter_disconnected and ('.' in data['smiles']): continue
            
            data['pos_ref'] = data['pos_ref'].reshape(-1, data['rdmol'].GetNumAtoms(), 3)
            data['pos_gen'] = data['pos_gen'].reshape(-1, data['rdmol'].GetNumAtoms(), 3)

            num_gen = data['pos_ref'].shape[0] * self.ratio
            if data['pos_gen'].shape[0] < num_gen: continue
            data['pos_gen'] = data['pos_gen'][:num_gen]

            filtered_data_list.append(data)

        filtered_data_list = filtered_data_list[start_idx:]
        self.print_fn('Filtered: %d / %d' % (len(filtered_data_list), len(packed_data_list)))

        covr_scores = []
        matr_scores = []
        covp_scores = []
        matp_scores = []
        for confusion_mat in tqdm(self.pool.imap(func, filtered_data_list), total=len(filtered_data_list)):
            # confusion_mat: (num_ref, num_gen)
            rmsd_ref_min = confusion_mat.min(-1)    # np (num_ref, )
            rmsd_gen_min = confusion_mat.min(0)     # np (num_gen, )
            rmsd_cov_thres = rmsd_ref_min.reshape(-1, 1) <= self.thresholds.reshape(1, -1)  # np (num_ref, num_thres)
            rmsd_jnk_thres = rmsd_gen_min.reshape(-1, 1) <= self.thresholds.reshape(1, -1) # np (num_gen, num_thres)

            matr_scores.append(rmsd_ref_min.mean())
            covr_scores.append(rmsd_cov_thres.mean(0, keepdims=True))    # np (1, num_thres)
            matp_scores.append(rmsd_gen_min.mean())
            covp_scores.append(rmsd_jnk_thres.mean(0, keepdims=True))    # np (1, num_thres)

        covr_scores = np.vstack(covr_scores)  # np (num_mols, num_thres)
        matr_scores = np.array(matr_scores)   # np (num_mols, )
        covp_scores = np.vstack(covp_scores)  # np (num_mols, num_thres)
        matp_scores = np.array(matp_scores)

        results = EasyDict({
            'CoverageR': covr_scores,
            'MatchingR': matr_scores,
            'thresholds': self.thresholds,
            'CoverageP': covp_scores,
            'MatchingP': matp_scores
        })
        # print_conformation_eval_results(results)
        return results
