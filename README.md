# GEFormer
  GEFormer is a genome-wide prediction model for genotype-environment interactions based on a deep learning approach designed to predict maize phenotypes using genotype and environment jointly.

# 0. Requirements
   We build GEFormer on the Python 3.9, pytorch1.8.1, We recommend anaconda environment for GEFormer.
# 1. Installation
   Clone this repository
   git clone https://github.com/Deep-Breeding/GEFormer
# 2. Build the virtual environment
   conda create -n GEFormer python=3.8     
   conda activate GEFormer               
   cd GEFomer                        
   pip install -r requirements.txt
# 3. Traing model
   python run_train.py
# 4. Testing phenotype
   python run_pred.py
# Contact
If you have any questions, please contact:liujianxiao@mail.hzau.edu.cn
