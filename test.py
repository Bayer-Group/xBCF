from src.xbcf import run_dict
import json


def test():
    """
    4 example SMILES are passed into the pipeline but only SMILES 1 and 2 
    will be computed due to the CDDD preprocessing.

    If you need them to be computed, specify the preproccessor as canonicalizer or None 
    """
    data_dict = {
            "smiles": ["Clc1ccccc1c2ccccc2Cl", "CCCCC(CC)COC(=O)c1ccc(C(=O)OCC(CC)CCCC)c(C(=O)OCC(CC)CCCC)c1", "CCCCCCCCCCC", "OC=O"],
            "ids": ["1", "fail_sml", "2", "3"],
            "method": "substitution",
            "only_predict": 'false',
            "preprocessor": "CDDD"
            }

    res = run_dict(data_dict)
    return res

if __name__ == "__main__":

    res = test()
    with open(f"./test.json", "w") as jf:
        json.dump(res, jf)
        
    print("Sensitivity scores have been generated successfully!")

