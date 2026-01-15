# Install the dependent environment

```bash
conda create -n GEFormer python=3.8

codna activate GEFormer

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install pandas einops scipy scikit-learn optuna
```


# Data Introduction

## CVF File Examples

Cross-validation files for splitting training and validation sets.

First column: ID, Second column: CV

M1_CVF.csv 【Ten fold cross validation】

| ID   | CV |
|------|----|
| ID1  | 7  |
| ID2  | 2  |
| ID3  | 1  |
| ID4  | 5  |
| ID5  | 9  |
| ID6  | 8  |
| ID7  | 7  |
| ID8  | 5  |
| ID9  | 8  |
| ID10 | 6  |
| ID11 | 10 |
| ID12 | 4  |
| ID13 | 9  |
| ID14 | 3  |
| ID15 | 5  |

M2_CVF.csv 【K-fold cross validation (K=number of environments)】

| ID      | CV |
|---------|----|
| ID1_HeB | 1  |
| ID2_HeB | 1  |
| ID3_HeB | 1  |
| ID4_HeB | 1  |
| ID5_HeB | 1  |
| ID6_HeB | 1  |
| ID7_HeB | 1  |
| ID8_HeB | 1  |
| ID9_HeB | 1  |
| ID10_HeB| 1  |
| ID11_HeB| 1  |
| ID12_HeB| 1  |
| ID13_HeB| 1  |
| ID14_HeB| 1  |
| ID15_HeB| 1  |
| ID1_LN  | 2  |
| ID2_LN  | 2  |
| ID3_LN  | 2  |
| ID4_LN  | 2  |
| ID5_LN  | 2  |
| ID6_LN  | 2  |
| ID7_LN  | 2  |
| ID8_LN  | 2  |
| ID9_LN  | 2  |
| ID10_LN | 2  |
| ID11_LN | 2  |
| ID12_LN | 2  |
| ID13_LN | 2  |
| ID14_LN | 2  |
| ID15_LN | 2  |

M3_CVF.csv 【Ten*K fold cross validation (K=number of environments)】

| ID      | CV |
|---------|----|
| ID1_HeB | 7  |
| ID2_HeB | 2  |
| ID3_HeB | 1  |
| ID4_HeB | 5  |
| ID5_HeB | 9  |
| ID6_HeB | 8  |
| ID7_HeB | 7  |
| ID8_HeB | 5  |
| ID9_HeB | 8  |
| ID10_HeB| 6  |
| ID11_HeB| 10 |
| ID12_HeB| 4  |
| ID13_HeB| 9  |
| ID14_HeB| 3  |
| ID15_HeB| 5  |
| ID1_LN  | 7  |
| ID2_LN  | 2  |
| ID3_LN  | 1  |
| ID4_LN  | 5  |
| ID5_LN  | 9  |
| ID6_LN  | 8  |
| ID7_LN  | 7  |
| ID8_LN  | 5  |
| ID9_LN  | 8  |
| ID10_LN | 6  |
| ID11_LN | 10 |
| ID12_LN | 4  |
| ID13_LN | 9  |
| ID14_LN | 3  |
| ID15_LN | 5  |

## Environment File Examples

Environmental data with fixed first two column names (env and date), followed by environmental factor data from the third column.

File Name：\${envName}_env.csv

LN_env.csv

| env | date       | DL     | GDD    | dGDD   | DTR    |
|-----|------------|--------|--------|--------|--------|
| LN  | 2014/5/11  | 14.371 | 6.147  | 0      | 7.758  |
| LN  | 2014/5/12  | 14.407 | 8.451  | 2.304  | 17.712 |
| LN  | 2014/5/13  | 14.442 | 13.896 | 5.445  | 31.014 |
| LN  | 2014/5/14  | 14.477 | 10.881 | 3.015  | 24.444 |
| LN  | 2014/5/15  | 14.512 | 10.584 | 0.297  | 26.19  |
| LN  | 2014/5/16  | 14.545 | 12.537 | 1.953  | 29.358 |
| LN  | 2014/5/17  | 14.578 | 13.635 | 1.098  | 27.756 |
| LN  | 2014/5/18  | 14.611 | 17.802 | 4.167  | 25.524 |
| LN  | 2014/5/19  | 14.643 | 13.248 | 4.554  | 16.416 |

## Genotype File Examples
Genotypic data with first column name: ID

geno.csv

| ID   | SNP1 | SNP2 | SNP3 | SNP4 | SNP5 | SNP6 | SNP7 | SNP8 | SNP9 | SNP10 |
|------|------|------|------|------|------|------|------|------|------|-------|
| ID1  | 0    | 0    | 0    | 0    | 1    | 0    | 0    | 1    | 0    | 0     |
| ID2  | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| ID3  | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| ID4  | 0    | 2    | 0    | 2    | 2    | 2    | 0    | 0    | 2    | 2     |
| ID5  | 0    | 0    | 0    | 0    | 0    | 0    | 2    | 2    | 0    | 0     |
| ID6  | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| ID7  | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| ID8  | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| ID9  | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| ID10 | 0    | 0    | 0    | 0    | 1    | 0    | 1    | 2    | 0    | 0     |
| ID11 | 2    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| ID12 | 0    | 2    | 0    | 2    | 2    | 2    | 0    | 0    | 2    | 2     |
| ID13 | 0    | 0    | 0    | 0    | 1    | 0    | 2    | 2    | 0    | 0     |
| ID14 | 2    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0     |
| ID15 | 0    | 0    | 2    | 0    | 2    | 0    | 1    | 1    | 0    | 0     |

## Phenotype File Examples
Phenotypic data

File Name：\${phenoName}_phe.csv

Column Name：\${phenoName}_\${envName}

PH_phe.csv

| ID   | PH_LN  | PH_HeB |
|------|--------|--------|
| ID1  | 241.13 | 228.8  |
| ID2  | 204    | 196.75 |
| ID3  | 207.4  | 233.33 |
| ID4  | 230.5  | 189.4  |
| ID5  | 200    | 183.75 |
| ID6  | 230.5  | 205.25 |
| ID7  | 211.6  | 225    |
| ID8  | 228    | 202.5  |
| ID9  | 248.2  | 195    |
| ID10 | 204    | 184.5  |
| ID11 | 205.4  | 192.5  |
| ID12 | 193.86 | 208    |
| ID13 | 239.6  | 211    |
| ID14 | 221.2  | 197.2  |
| ID15 | 161.8  | 138.2  |



# Parameter Introduction

## Required Parameters
- `--geno_path` // Path to genotype data file
- `--phe_folder` // Folder containing phenotype data
- `--pheno_name` // Phenotype name
- `--env_folder` // Folder containing environment data
- `--env_name` // Environment name required for M1 and M3 schemes
- `--cvf_folder` // Folder for training/validation set configuration
- `--scheme` // Choose M1/M2/M3 scheme
- `--device` // Which GPU to use
- `--need_optuna` // Whether to enable hyperparameter optimization

## Hyperparameters
When `--need_optuna` is True: Automatically optimize combinations of the following hyperparameters within specified ranges

When `--need_optuna` is False: Use default parameters or specify the following parameters

### Basic Parameters
- `--batch` // Batch size (default: 64)
- `--dropout` // Dropout rate (prevents overfitting) (default: 0.3)
- `--depth` // Number of feature layers (default: 2)
- `--neurons1` // Number of neurons 1 (default: 256)
- `--neurons2` // Number of neurons 2 (default: 32)
- `--lr` // Learning rate (default: 5e-4)

### Optuna Optimization Parameters
- `--need_optuna` // Whether to enable hyperparameter optimization (default: True)
- `--optuna_epoch` // Number of attempts with different parameter combinations (default: 100)
- `--num_fold` // Folded number (default: 1)

### Optuna Search Ranges
- `--batch_1` // Minimum batch size (default: 16)
- `--batch_2` // Maximum batch size (default: 128)
- `--dropout_1` // Minimum dropout rate (default: 0.2)
- `--dropout_2` // Maximum dropout rate (default: 0.6)
- `--depth_1` // Minimum depth (default: 1)
- `--depth_2` // Maximum depth (default: 6)
- `--neurons1_1` // Minimum neurons1 number (default: 128)
- `--neurons1_2` // Maximum neurons1 number (default: 512)
- `--neurons2_1` // Minimum neurons2 number (default: 1)
- `--neurons2_2` // Maximum neurons2 number (default: 128)
- `--lr_1` // Minimum learning rate (default: 1e-7)
- `--lr_2` // Maximum learning rate (default: 1e-2)

# Usage Examples

```bash
python train.py \
    --geno_path ../data/geno/geno.csv \
    --phe_folder ../data/phe \
    --pheno_name PH \
    --env_folder ../data/env \
    --env_name LN \
    --cvf_folder ../data/cvf \
    --scheme M1 \
    --device cuda:0
```

