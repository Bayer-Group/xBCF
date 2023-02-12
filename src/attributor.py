import os
import re
import numpy as np
import pandas as pd
from random import randint
import pickle
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import sys
sys.path.append('./')
from cddd.inference import InferenceModel

use_gpu = False
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

char_dict = {
    0: '</s>',
    1: '#',
    2: '%',
    3: ')',
    4: '(',
    5: '+',
    6: '-',
    7: '1',
    8: '0',
    9: '3',
    10: '2',
    11: '5',
    12: '4',
    13: '7',
    14: '6',
    15: '9',
    16: '8',
    17: ':',
    18: '=',
    19: '@',
    20: 'C',
    21: 'B',
    22: 'F',
    23: 'I',
    24: 'H',
    25: 'O',
    26: 'N',
    27: 'P',
    28: 'S',
    29: '[',
    30: ']',
    31: 'c',
    32: 'i',
    33: 'o',
    34: 'n',
    35: 'p',
    36: 's',
    37: 'Cl',
    38: 'Br',
    39: '<s>'
    }


class Attributor:
    """
    The attributor class
    """
    def __init__(self, qsar_model_path, cddd512_model_dir=None, cpu_threads=2):
        # Use option I to install CDDD
        if cddd512_model_dir is not None:
            self.cddd_model = InferenceModel(cddd512_model_dir,
                                             use_gpu,
                                             cpu_threads=cpu_threads)
        # Use option II to install CDDD
        else:
            self.cddd_model = InferenceModel(use_gpu=use_gpu,
                                             cpu_threads=cpu_threads)
        self.qsar_mdl = pickle.load(open(qsar_model_path, 'rb'))

    def smiles_attribution(self, smiles, ids=None, method="substitution", plot=True):
        """The main interface for providing atom-wise and char-wise attributions


        Parameters
        ----------
        smiles : str or list of str
            The input SMILES strings (single or list), if it is a single string, the results will be
            printed or plotted to the screen. If it is a list, the results will be returned as a
            dict for being stored as json files, which can also be viewed by XSMILES.
        ids : list
            A list of identifiers (e.g. the name of the compounds) when smiles is a list. If
            provided, it should have the same length as smiles
        method : str, optional
            either "substitution" or "deletion", by default "substitution"
        plot : bool, optional
            If True, it plots the atom attributions.
 
        Returns
        -------
        pandas dataframe
            return the attribution scores for each character in the input SMILES
        """

        if isinstance(smiles, str):
 
            self._smiles_parser(smiles)
            attributions = self._get_attributions(smiles, method=method)
            chars = list(char_dict.values())[1:-1]
            chars_df = pd.concat(
                [
                    pd.DataFrame(self.char_list, columns=['Char']),
                    pd.DataFrame(attributions["bcf_attributions"], columns=['BCF_score']),
                ], axis=1)
 
            atom_bcf_scores = np.array(attributions["bcf_attributions"])[self.atom_char_list]
 
            atom_df = pd.concat(
                [
                    pd.DataFrame(self.true_char_list, columns=['Atom']),
                    pd.DataFrame(atom_bcf_scores, columns=['bcf_scores']),
                ], axis=1)
 
            special_atom_dict = self._get_special_dict(attributions)
            if special_atom_dict:
                special_atom_df = pd.DataFrame.from_dict(special_atom_dict)
                print(f"Special atoms dataframe: {special_atom_df}")
 
            if plot:
                print(f"atom_id: {atom_df}")
                print(f"char_id: {chars_df}")
 
                mol = Chem.MolFromSmiles(smiles)
                print("BCF sensitivity scores: ")
                fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_bcf_scores)
            return chars_df
 
        elif isinstance(smiles, list):
 
            sml_attr = []
            special_atom_dfs = []
            for i, sml in enumerate(smiles):
                print(f"Processing {sml}")
                self._smiles_parser(sml)
                attributions = self._get_attributions(sml, method=method)

                bcf_attr = attributions['bcf_attributions']

                attributes_dict = {
                    "attributes":
                    {
                        # compound ID: e.g. compound name
                        "pred_logBCF": attributions['pred_logBCF'],
                        f"max_logBCF_sens_{method}": np.max(np.abs(bcf_attr)),
                    }
                }

                # IF ids are given then update the attribute dict
                if ids is not None:
                    assert len(ids) == len(smiles)
                    attributes_dict['attributes'].update({"Compound ID": ids[i]})
                res_dict = {
                        "string": sml,
                        "sequence": attributions['chars'],
                        "methods": [
                            {"name": f"logBCF {method} sensitivity score", "scores": bcf_attr},
                        ],
                        "attributes": attributes_dict['attributes']
                        }
                sml_attr.append(res_dict)
            return sml_attr

        else:
            raise ValueError("The input SMILES must be a string or a list of strings.")

    def _smiles_parser(self, sml):
        """Parse the input smiles to get list of chars or atom
        """
        REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
        self.atom_chars = ["Cl","Br","C","B","F","I","H","O","N","P","S","c","i","o","n","p","s"]
        self.char_list = re.findall(REGEX_SML, sml)
        self.atom_char_list = [i for i in range(len(self.char_list)) if self.char_list[i] in self.atom_chars]
        self.true_char_list = [self.char_list[i] for i in self.atom_char_list]

    def _get_special_dict(self, attributions):
        subset_key = ['special_char', 'sens_special_lod',
                      'sens_special_bcf', 'special_char_idx',
                      'special_atom_idx', 'SMILES']
        special_atom_dict = {k: attributions[k] for k in subset_key if k in attributions}
        return special_atom_dict

    def _get_attributions(self, sml, method='substitution'):
        """Get character sensitivity scores for the given SMILES
        Two ways of getting scores:
            1. deletion: delete the on-position value
            2. substitution: flip the off-position values one by one and get the average
 
        The flipping idea: what is the expected prediction change if the right token
        is replaced with any other token.
 
        1. Compute original BCF values
        2. for each atom:
            a. Generate fake smiles strings for every character in the vocab
            b. Get embeddings for those smiles
            c. Compute BCF values according to embeddings
            d. Average BCF values
            g. Compute the difference as the token attribution
            h. Return token attributions
 
        The deletion method is to estimate the effect of absence of certain atoms
 
        1. Compute original logBCF
        2. for each atom:
            a. Pass its index to cddd model creator for deleting the atom
            b. Get the CDDD embeddings
            c. Pass embeddings to QSAR model
 
 
        Paramters
        -----------
        sml : str
            Specify which method to substitution SMILES
        method : str
            Specify which method to substitution SMILES
 
        Returns
        -------
        dict
            A dict with sensitivity scores for each position and predictions 
        """
        emb = self.cddd_model.seq_to_emb(sml)
        orig_y = self.qsar_mdl.predict(emb)[0]
 
        char_vocab = list(char_dict.values())[1:-1]
 
        chars = self.char_list
 
        attributions = {
            "bcf_attributions": [],
            "chars": chars,
            "pred_logBCF": orig_y
            }
 
        if method == "substitution":
            for i, sml_char in enumerate(chars):
                sml_copy = chars
                mutated_smls = ["".join(sml_copy[:i] + [w] + sml_copy[i+1:]) for w in char_vocab]
                mutated_emb = self.cddd_model.seq_to_emb(mutated_smls)
                y = self.qsar_mdl.predict(mutated_emb)
                mutated_bcf = np.mean(y)
                attributions['bcf_attributions'].append(orig_y - mutated_bcf)
 
            print(f"The predicted logBCF={orig_y}")

            # print(f"The ground truth logBCF={self.bcf}")
        elif method == "deletion":
            for i, sml_char in enumerate(chars):
                sml_copy = chars
                sml_copy = "".join(sml_copy[:i] + ["A"] + sml_copy[i+1:])
                masked_emb = self.cddd_model.seq_to_emb("".join(sml_copy))
                masked_y = self.qsar_mdl.predict(masked_emb)
                attributions['bcf_attributions'].append(orig_y - masked_y)
        # Get the attributions values for the special chars for comparing with correction factors in EPWIN
        special_chars = ['Br', 'Cl', 'I', 'n']
        special_char_idx = [i for i in range(len(chars)) if chars[i] in special_chars]
        # update the attributions dict when there is special atoms
        if special_char_idx:
            special_char_list = [chars[i] for i in special_char_idx]
            special_atom_idx = [i for i in range(len(self.true_char_list)) if self.true_char_list[i] in special_chars]
            attr_special_bcf = np.array(attributions['bcf_attributions'])[special_char_idx]
            attributions.update({
                'special_char': special_char_list,
                'attr_special_bcf': attr_special_bcf,
                'special_char_idx': special_char_idx,
                'special_atom_idx': special_atom_idx,
                'SMILES': sml
            })
        return attributions

    def predict(self, sml, return_emb=False, ids=None):
        """
        Interface for predicting logBCF for the given SMILES

        Parameters
        ----------
        sml: str or list of str
            The SMILES of interest for prediction
        return_emb: bool
            If return the generated embedding for the given sml
        ids: str
            Compound IDs for each prediction
        """
        new_emb = self.cddd_model.seq_to_emb(sml)
        bcf = self.qsar_mdl.predict(x)

        if ids is not None:
            res = {
                    "Compound ID": ids,
                    "smiles": sml,
                    'logBCF_pred': bcf
                    }
        else:
            res = {
                    "smiles": sml,
                    'logBCF_pred': bcf
                    }
        if return_emb:
            return res, new_emb
        else:
            return res


def rdkit_canonicalizer(smiles):
    """
    the input smiles can be a single SMILES string, a list of SMILES or a Pandas dataframe
    with the column name as "SMILES"
    """
    if isinstance(smiles, pd.DataFrame):
        for row in smiles.itertuples():
            mol = Chem.MolFromSmiles(row.SMILES)
            try:
                rdkit_sml = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                smiles.loc[row.Index, 'SMILES_rdkit'] = rdkit_sml
            except Exception as err:
                print(err)
                print(row.SMILES, row.Compound_No)
        return smiles
    elif isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        rdkit_sml = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return rdkit_sml
    elif isinstance(smiles, list):
        canon_smiles = []
        for sml in smiles:
            mol = Chem.MolFromSmiles(sml)
            canon_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True))
        return canon_smiles
    else:
        raise ValueError("Unknown input type!")
