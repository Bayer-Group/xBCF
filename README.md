# xBCF - Predicting bioconcentration factor with explainable deep learning

Open source project for predicting logBCF with explanations on SMILES level

## SMILES preprocessing
Make sure your input SMILES strings fulfill the CDDD requirements, i.e. mono-constituent organic molecules, desalted, and neutralized.

## Usage

The code has been tested with Python 3.7.

Core dependencies:
- Tensorflow 1.13.0

```bash
python3.7 -m venv bcf_env
source ./bcf_env/bin/activate
pip install -r requirements.txt
python test.py
```

If everything goes well, the last step should print the attribution and predictions for two SMILES.


## Dependency on CDDD
### Option I - customized way of using the official CDDD repo

1. Place the CDDD source code under `./src`
2. Place the CDDD model directory under './models'


### Option II - an unofficial package

In this sister [repo](https://github.com/Bayer-Group/xsmiles-use-cases), an official `.whl` file was packed with the CDDD source code and the default CDDD model. 
Please refer to the repo for installation. And use this installation method at your risk. 

The script `src/attributor.py` works with both ways of setting up CDDD, make sure you modify `src/xbcf.py` accordingly.



