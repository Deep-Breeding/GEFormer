import argparse
from train import geformer

parser = argparse.ArgumentParser(description="Genome-wide prediction model for genotype-environment interaction.")

parser.add_argument('--geno_path', type=str, default="./data/data_train/geno.csv", help='path of geno file')
parser.add_argument('--pheno_path', type=str, default="./data/data_train/phe.csv", help='path of pheno file')
parser.add_argument('--pheno_name', type=str, default="PH", help='name of phenotype')
parser.add_argument('--env_path', type=str, default="./data/data_train/env.csv", help='path of environment file')
parser.add_argument('--CVF_path', type=str, default="./data/data_train/CVF.csv", help='path of cvf file')
parser.add_argument('--model_path', type=str, default="./model", help='path of model file')
parser.add_argument('--device', default="cuda:2", help='device id (i.e. 0 or 0,1 or cpu)')

parser.add_argument("--optuna", type=lambda x: x.lower() == "true", default=False, help='whether to adjust parameters')

parser.add_argument('--optuna_epoch', type=int,default=10, help='number of attempts with different parameter combinations')
parser.add_argument('--batch', type=int, default=161, help='batchsize')
parser.add_argument('--dropout', type=float, default=0.41, help='dropout')
parser.add_argument('--depth', type=int, default=2, help='depth')
parser.add_argument('--neurons1', type=int, default=607, help='neurons1 number')
parser.add_argument('--neurons2', type=int,default=250, help='neurons2 number')
parser.add_argument('--lr', type=float,default=0.0007796, help='learning rate')

args = parser.parse_args()

geformer(args.geno_path, args.pheno_path, args.pheno_name, args.env_path, 
         args.CVF_path, args.model_path, args.device, args.optuna, 
         args.optuna_epoch, args.batch, args.dropout, args.depth, 
         args.neurons1, args.neurons2, args.lr) 

#python run_train.py --geno_path ./data/data_train/geno.csv --pheno_path ./data/data_train/phe.csv --pheno_name PH --env_path ./data/data_train/env.csv --CVF_path ./data/data_train/CVF.csv --model_path ./model --device cuda:0 --optuna False --depth 11 --batch 11

#python run_train.py --geno_path ./data/data_train/geno.csv --pheno_path ./data/data_train/phe.csv --pheno_name PH --env_path ./data/data_train/env.csv --CVF_path ./data/data_train/CVF.csv  --model_path ./model  --device cuda:0 --optuna True