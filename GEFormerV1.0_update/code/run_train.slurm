#!/bin/bash
#SBATCH -J GEF
#SBATCH -p 5090-0
#SBATCH -N 1

SCHEME=M1
EnvName=LN
PhenoName=PH
OUTFILE=out-${EnvName}-${PhenoName}_${SCHEME}.txt
ERRFILE=err-${EnvName}-${PhenoName}_${SCHEME}.txt

python train.py \
--geno_path '../data/geno/geno.csv' \
--phe_folder '../data/phe' \
--pheno_name ${PhenoName} \
--env_folder '../data/env' \
--env_name ${EnvName} \
--cvf_folder '../data/cvf' \
--scheme ${SCHEME} \
--device cuda:0 \
> $OUTFILE 2> $ERRFILE
