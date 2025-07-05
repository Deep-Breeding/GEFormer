# GEFormer V1.0
 GEFormer: A genotype-environment interaction-based genomic prediction method that integrates the gating multilayer perceptron and linear attention mechanisms

![GEFormer](imgs/GEFormer.png)  

##  Operation systems

    Windows

    Linux
   
    MacOS

##  Requirements
    
    Python3.8
 
    Pytorch1.8.1(at least)

## Installation

### 1  The first installation method by using git clone.

(1) git clone: https://github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0
   
    or: https://pan.baidu.com/s/1HrJSrV_tPrsqmllEHEmYqA (Extraction code: 1234)

(2) Build the virtual environment.

    conda create -n GEFormer python=3.8  
   
    conda activate GEFormer
               
    cd GEFomerV1.0
                       
    pip install -r requirements.txt

### 2  The second installation method by using docker.
    
    docker: https://hub.docker.com/r/coder02lq/geformer

    docker pull coder02lq/geformer:v1

### 3  The third installation method by using PiP.

    conda create -n GEFormer python=3.8  
   
    conda activate GEFormer

    pip install GEFormerV1.0

## User Manual

1 git clone user manual： [https://github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0/GEFormerV1_ usermannual.pdf](https://github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0/GEFormerV1_ usermannual.pdf)

2 docker user manual(Chinese):  [https://github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0_docker/GEFormerV1_ docker_usermannual.pdf](https://github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0_docker/GEFormerV1_ docker_usermannual.pdf)

## Input data

Genotype data; Phenotype data; environment data; Data partitioning

1 GEFormerV1.0 data: [*https: //github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0/data*](https: //github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0/data)

2 GEFormerV1.0_docker data: [*https: //github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0_docker/data*](https: //github.com/Deep-Breeding/GEFormer/tree/main/GEFormerV1.0_docker/data)

## Training model

The example of training model:

   python run_train.py --geno_path ./data/geno.csv --pheno_path ./data/phe.csv --pheno_name PH --env_path ./data/env.csv --CVF_path ./data/CVF.csv  --model_path ./model  --device cpu --optuna True

Parameters:

|                 |    Description  |
|----------------:|-------------|
|    geno_path        |    Genotype file path   |
|    pheno_path       |    Phenotype file path    |
|    pheno_name   |    Phenotype name    |
|    env_path |     Environment file path    |
|    CVF_path    |    Set up training and test sets     |
|    model_path         |    Output file path   |  
|    device    |    Runing device (CPU or GPU)     |
|    optuna         |    hyperparameter optimization   |  


Output:
  
  After training, the Pearson correlation coefficient between the predicted values and the true values is written in the log file, as shown as the following: 

  Pearson = (0.71, 0.0030)

  The value is the Pearson correlation coefficient (0.71) and the second number is the P-value (0.0030). 

## Citation

You can read our paper explaining GEFormer.

Yao Z, Yao M, Wang C, et al. (2025). GEFormer: A genotype-environment interaction-based genomic prediction method that integrates the gating multilayer perceptron and linear attention mechanisms. Molecular Plant, 2025, 18(3): 527-549. https://doi.org/10.1016/j.molp.2025.01.020

## Contact
If you have any questions, please contact:liujianxiao@mail.hzau.edu.cn
