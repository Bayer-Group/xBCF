import os
import numpy as np
import pandas as pd
import pickle
import sys
import json
import argparse
from cddd.preprocessing import preprocess_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import re
from src.attributor import Attributor, rdkit_canonicalizer

############################################

QSAR_PATH = "../models/final_cddd512_rdkit_svr.pkl"
QSAR_PATH = os.path.join(os.path.dirname(__file__), QSAR_PATH)
# CDDD_DIR = "../models/cddd_default_model"
# CDDD_DIR = os.path.join(os.path.dirname(__file__), CDDD_DIR)
############################################


def run_dict(json_data):
    """Generate sensitivity scores for any input SMILES using substitution method,
    the method can be modified to deletion.
    Parameters
    ----------
    json_data: dict
        Input data containing the SMILES to be predicted, which bears the
        format of {
            "smiles": list of smiles
            'ids'   : list of ids
            'only_predict': "false" or "true"
            'preprocessor': "CDDD" or "canonicalizer" or "" (empty)
            'method': "substitution" or "deletion"
        }
        ids must be mapped with smiles! only_predict is set to be string
        to cope with deployment on different platforms, where native python
        Boolean may not be supported. Both preprocessors are based on RDKit 
        canonicalization, but CDDD has additional filter criteria based on
        AppDomain of CDDD. If preprocessor is undefined, no structure standardization
        is carried out.

    return
    ------
    dict
        A dictionary with predefined structure compatible with XSMILES
    """
    if isinstance(json_data['smiles'], list):
        if json_data["preprocessor"] == "CDDD":
            smls = [preprocess_smiles(s) for s in json_data['smiles']]  # rdkit_canonicalizer(smiles)
            smls = [str(s) for s in smls]
            idx_nan = [i for i in range(len(smls)) if smls[i]=="nan"]
            if 'ids' in json_data:
                ids = json_data['ids']
                assert len(ids) == len(smls)
                if len(idx_nan) != 0:
                    unprocessable_ids = [ids[i] for i in idx_nan]
                    smls = [j for i, j in enumerate(smls) if i not in idx_nan]
                    ids = [j for i, j in enumerate(ids) if i not in idx_nan]
                    print("Please note that the following molecules were not processed", unprocessable_ids)
            else: ids = None
        elif json_data["preprocessor"] == "canonicalizer":
            smls = []
            for s in json_data['smiles']:
                try:
                    smls.append(str(rdkit_canonicalizer(s)))
                except:
                    smls.append('nan')
            idx_nan = [i for i in range(len(smls)) if smls[i]=="nan"]
            if 'ids' in json_data:
                ids = json_data['ids']
                assert len(ids) == len(smls)
                if len(idx_nan) != 0:
                    unprocessable_ids = [ids[i] for i in idx_nan]
                    smls = [j for i, j in enumerate(smls) if i not in idx_nan]
                    ids = [j for i, j in enumerate(ids) if i not in idx_nan]
                    print("Please note that the following molecules were not processed", unprocessable_ids)
            else: ids = None
        else:
            smls = json_data["smiles"]
            if 'ids' in json_data:
                ids = json_data['ids']
                assert len(ids) == len(smls)
            else: ids = None

        print(f"Number of SMILES after preprocessing: {len(smls)}")
        # install CDDD with option I
        # mtl_attributor = Attributor(QSAR_PATH, CDDD_DIR)
        # install CDDD with option II
        mtl_attributor = Attributor(QSAR_PATH)

        if json_data['only_predict'] == 'true':
            res = mtl_attributor.predict(smls, ids=ids)
        elif json_data['only_predict'] == 'false':
            if json_data['method'] in ['substitution', 'deletion']:
                method = json_data['method']
            else:
                method = 'substitution'
            res = mtl_attributor.smiles_attribution(smls, ids=ids, method=method)
        else:
            res = f"Not known input only_predict = {json_data['only_predict']}"
    else:
        res = "Oooops...a List of SMILES is expected!"

    return res


if __name__ == "__main__":

    smile_list = ['COC(=O)c1sccc1S(=O)(=O)NC(=O)Nc2nc(C)nc(OC)n2', '[Au]c1sccc1S(=O)(=O)NC(=O)Nc2nc(C)nc(OC)n2']
    id_list = ['1', '2']

    data_dict = {'smiles': smile_list,
                 'ids': id_list,
                 'only_predict': 'true',
                 'method': 'substitution',
                 'preprocessor': 'canonicalizer'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Input config", default=data_dict)
    args = parser.parse_args()

    #res = run(args.smiles, args.onlypredict)
    #res = run(smile_list)
    #print("Result from run:", res)

    res = run_dict(args.data)
    print("Result from run_dict only predict:", res)
    res = run_dict({'smiles': smile_list, 'ids': id_list, 'only_predict': 'false', 'method': 'deletion', 'preprocessor': 'CDDD'})
    print("Result from run_dict:", res)


