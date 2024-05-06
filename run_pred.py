import argparse
from pred import predict


parser = argparse.ArgumentParser(description="Genome-wide prediction model for genotype-environment interaction.")

parser.add_argument('--geno_path', type=str, default="./data/data_test/test_geno.csv", help='path of genotype file')
parser.add_argument('--env_path', type=str, default="./data/data_test/test_env.csv", help='path of environment file')
parser.add_argument('--pheno_name', type=str, default="PH", help='name of phenotype')
parser.add_argument('--model_path', type=str, default="./model/bestmodel.pkl", help='path of model file')
parser.add_argument('--result_path', type=str, default="./result/Predict.csv", help='path of result file')
parser.add_argument('--device', default="cuda:2", help='device id (i.e. 0 or 0,1 or cpu)')

args = parser.parse_args()

predict(args.geno_path, args.env_path, args.pheno_name, args.model_path, args.result_path, args.device)
