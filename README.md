# GEFormer V1.0
  GEFormer is a genome-wide prediction model for genotype-environment interactions based on a deep learning approach designed to predict maize phenotypes using genotype and environment jointly.

# 0. Requirements
   We build GEFormer on the Python 3.8, pytorch1.8.1, We recommend anaconda environment for GEFormer.
# 1. Installation
   Clone this repository
   git clone https://github.com/Deep-Breeding/GEFormer
   or https://pan.baidu.com/s/1HrJSrV_tPrsqmllEHEmYqA (Extraction code: 1234)
# 2. Build the virtual environment
   conda create -n GEFormer python=3.8     
   conda activate GEFormer               
   cd GEFomer                        
   pip install -r requirements.txt
# 3. Input data file
(1) Genotype file: geno.csv
(2) Phenotype file: phe.csv
(3) Environment file: env.csv
(4) Data partitioning: CVF.csv
# 4. Traing model
   Parameters:
    --geno_path   // Genotype file path
    --pheno_path  // Phenotype file path
    --pheno_name  // Phenotype name
    --env_path    // Environment file path
    --CVF_path   // Set up training and validation sets
    --model_path  // Output file path
    --device      //Runing device (CPU or GPU)
    -- optuna     // hyperparameter optimization
    The following are optional parameters:
      -- optuna_epoch   // Number of tuning
      --batch          //batch size
      --lr             // learn reating
      --drop_out       //drop put
      --depth          //feature depth
      --neurons1       // neurons number 1
      --neurons2       // neurons number 2

  The example of training model:
  python run_train.py --geno_path ./data/geno.csv --pheno_path ./data/phe.csv --pheno_name PH --env_path ./data/env.csv --CVF_path ./data/CVF.csv  --model_path ./model  --device cpu --optuna True
  Output:
  After training, the Pearson correlation coefficient between the predicted values and the true values is written in the log file, as shown as the following: 
  Pearson = (0.71, 0.0030)
  The value is the Pearson correlation coefficient (0.71) and the second number is the P-value (0.0030). 

# Contact
If you have any questions, please contact:liujianxiao@mail.hzau.edu.cn
