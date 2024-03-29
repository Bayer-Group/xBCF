# xBCF - Predicting bioconcentration factor with explainable deep learning

Open source project for predicting logBCF with explanations on SMILES level.
Find more detailed descriptions of the methodology in our research article: https://doi.org/10.1016/j.ailsci.2022.100047

## SMILES preprocessing
Make sure your input SMILES strings fulfill the CDDD requirements, i.e. mono-constituent organic molecules, desalted, and neutralized.

## Dependency on CDDD

The models in the project are built upon CDDD embeddings which depends on TF 1.X. The latest Python version supported TF1.X is 3.7. We have adjust all code dependencies to Python 3.7 to better fit different deployment platforms. Therefore, we suggest the following ways to install the CDDD package.  

### Option I - customized way of using the official CDDD repo

1. Place the CDDD source code under `./src`
2. Place the CDDD model directory under `./models`


### Option II - an unofficial package

In this sister [repo](https://github.com/Bayer-Group/xsmiles-use-cases), an unofficial `.whl` file was packed with the CDDD source code and the default CDDD model. 
Please refer to the repo for installation. And use this installation method at your own risk. 

The script `src/attributor.py` works with both ways of setting up CDDD, make sure you modify `src/xbcf.py` accordingly.

For any questions, feel free to open issues.

## Model availability
Our MTL model as reported in our paper cannot be published due to the inclusion of internal secret data in the training set. 
Nevertheless, an SVR model with good performance is published in the repo. And a linear regression model is also available as a simple baseline. 

## Usage

The code has been tested with Python 3.7 

Core dependencies:
- Tensorflow 1.13.0

```bash
python3.7 -m venv bcf_env
source ./bcf_env/bin/activate
pip install -r requirements.txt
python test.py
```
If everything goes well, the last step should print the attribution and predictions for two SMILES.




## Please cite

`Zhao, L., Floriane, M., Heberle, H., & Schmidt, S. (2022). Modeling bioconcentration factors in fish with explainable deep learning. Artificial Intelligence in the Life Sciences, 2, 100047. https://doi.org/10.1016/j.ailsci.2022.100047`

```bibtex
@article{Zhao_Modeling_bioconcentration_factors_2022,
author = {Zhao, Linlin and Floriane, Montanari and Heberle, Henry and Schmidt, Sebastian},
doi = {10.1016/j.ailsci.2022.100047},
journal = {Artificial Intelligence in the Life Sciences},
month = {December},
pages = {100047},
title = {{Modeling bioconcentration factors in fish with explainable deep learning}},
volume = {2},
year = {2022}
}
```




