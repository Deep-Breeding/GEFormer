import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import numpy as np
import csv
import pandas as pd
import os
from torch.utils.data import DataLoader
from itertools import islice
import random
import argparse
import copy
import time
from sklearn.preprocessing import StandardScaler

from GEFormer import GEFormer


from dataset_new import myDataset

import optuna
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
setup_seed(147)

parser = argparse.ArgumentParser(description="Genome-wide prediction model for genotype-environment interaction.")

parser.add_argument('--geno_path', type=str, default='../data/geno/geno.csv', help='path of geno file')
parser.add_argument('--phe_folder', type=str, default='../data/phe/', help='path of phe folder')
parser.add_argument('--pheno_name', type=str, default='EW', help='name of phenotype')
parser.add_argument('--env_folder', type=str, default='../data/env/', help='path of env folder')
parser.add_argument('--env_name', type=str, default='JL', help='name of env')
parser.add_argument('--cvf_folder', type=str, default='../data/cvf/', help='path of cvf folder')
parser.add_argument('--scheme', type=str, default='M1', help='Three options')

parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

parser.add_argument('--batch', type=int, default=64, help='batchSize')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--depth', type=int, default=2, help='depth')
parser.add_argument('--neurons1', type=int, default=256, help='neurons1 number')
parser.add_argument('--neurons2', type=int, default=32, help='neurons2 number')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

parser.add_argument('--need_optuna', type=str2bool, default=True, help='need to adjust parameters')
parser.add_argument('--optuna_epoch', type=int, default=100,
                    help='number of attempts with different parameter combinations')
parser.add_argument('--num_fold', type=int, default=1, help='Folded number')
parser.add_argument('--batch_1', type=int, default=16, help='min_batchSize')
parser.add_argument('--batch_2', type=int, default=128, help='max_batchSize')
parser.add_argument('--dropout_1', type=float, default=0.2, help='min_dropout')
parser.add_argument('--dropout_2', type=float, default=0.6, help='max_dropout')
parser.add_argument('--depth_1', type=int, default=1, help='min_depth')
parser.add_argument('--depth_2', type=int, default=6, help='max_depth')
parser.add_argument('--neurons1_1', type=int, default=128, help='The minimun of the neurons1 number')
parser.add_argument('--neurons1_2', type=int, default=512, help='The maximun of the neurons1 number')
parser.add_argument('--neurons2_1', type=int, default=1, help='The minimun of the neurons2 number')
parser.add_argument('--neurons2_2', type=int, default=128, help='The maximun of the neurons2 number')
parser.add_argument('--lr_1', type=float, default=1e-7, help='min_learning rate')
parser.add_argument('--lr_2', type=float, default=1e-2, help='max_learning_rate')

args = parser.parse_args()
scheme = args.scheme


t1 = time.perf_counter()

list_number = []
list_phe = []
dictSeq = {}  # key: id+env_name   value:seq
dictSeq_1 = {}  # M2_M3 key: id+env_name    value: env

data_E = pd.DataFrame()  # M1  env
dict_env = {}  # M2_M3 key: env_name    value: env

scaler = StandardScaler()


# Processing environmental data
env_days = 0
env_factor = 0
def handle_env(env_path):
    hjenv = open(env_path, 'r')
    df_data = pd.read_csv(hjenv)
    df_data = df_data.iloc[:, 1:]
    time_data = df_data.iloc[:, 0]
    env_data = df_data.iloc[:, 1:]
    scaler.fit(env_data)
    env_data = scaler.fit_transform(env_data)
    env_data = pd.DataFrame(env_data)
    env_days = df_data.shape[0]
    env_factor = env_data.shape[1]
    return pd.concat([time_data, env_data], axis=1), env_days, env_factor

csv_env_files = [f for f in os.listdir(args.env_folder) if f.endswith('.csv')]
env_num = len(csv_env_files)
if scheme == 'M1':
    env_path = os.path.join(args.env_folder, args.env_name + '_env.csv')
    data_E, env_days, env_factor = handle_env(env_path)
else:
    for file in csv_env_files:  # Read files from the env folder
        env_path = os.path.join(args.env_folder, file)
        f_name = file.split('.')[0]  # xxx_env
        dict_env[f_name], env_days, env_factor = handle_env(env_path)

args.enc_in = env_factor
args.c_out = env_factor


# Processing genotype data
with open(args.geno_path) as file:
    reader = csv.reader(file)
    first_row = next(reader)
    snp_len = len(first_row) - 1

def handle_geno(geno_path, name=None):
    env_name = '_'+name.split('_')[0] if name is not None else ''
    with open(geno_path) as file:
        for line in islice(file, 1, None):
            num = line.split(",")[0] + env_name  # id+env_name
            list_str = line.split(",")[1:]
            dictSeq[num] = [int(x) for x in list_str]
            if name is not None:
                dictSeq_1[num] = dict_env[name]

if scheme == 'M1':
    handle_geno(args.geno_path)
else:
    for f in csv_env_files:
        f_name = f.split('.')[0]  # xxx_env
        handle_geno(args.geno_path, f_name)


# Processing phenotype data
def handle_phe(phe_path, env_name=args.env_name):
    if scheme != 'M1':
        env_name = env_name.split('_')[0]
    with open(phe_path) as file:
        for line in islice(file, 1, None):
            num = line.split(",")[0] + '_' + env_name if scheme != 'M1' \
                else line.split(",")[0]
            if num in dictSeq.keys():
                list_number.append(num)  # ID list
                item_phe = line.split(",")[idx[env_name]]
                list_phe.append(float(item_phe))

phe_path = os.path.join(args.phe_folder, args.pheno_name+'_phe.csv')
idx = {}  # { env: index }
with open(phe_path) as file:
    reader = csv.reader(file)
    first_row = next(reader)
    for index, el in enumerate(first_row[1:]):
        idx[el.split('_')[1]] = index + 1
if scheme == 'M1':
    handle_phe(phe_path)
else:
    for f in csv_env_files:
        f_name = f.split('.')[0]
        handle_phe(phe_path, f_name)


# cvf文件
cvf_path = os.path.join(args.cvf_folder, scheme+'_CVF.csv')
dt = open(cvf_path, 'r')
df = pd.read_csv(dt)



# Divide training and validation
def split_data(num_fold):
    list_number2 = []
    list_phe2 = []
    dictSeq2 = {}
    dictSeq2_1 = {}

    list_number1 = []
    list_phe1 = []
    dictSeq1 = {}
    dictSeq1_1 = {}

    val_data = df[(df['CV'] == num_fold) & df['ID'].str.endswith(args.env_name)].index if scheme == 'M3' \
        else df[df['CV'] == num_fold].index
    train_data = df[(df['CV'] != num_fold) & ~df['ID'].str.endswith(args.env_name)].index if scheme == 'M3' \
        else df[(df['CV'] != num_fold)].index


    for h3 in range(int(len(train_data))):
        list_number2.append(list_number[train_data[h3]])
        list_phe2.append(list_phe[train_data[h3]])
        dictSeq2[list_number2[h3]] = dictSeq[list_number2[h3]]
        if scheme != 'M1':
            dictSeq2_1[list_number2[h3]] = dictSeq_1[list_number2[h3]]

    for h2 in range(int(len(val_data))):
        list_number1.append(list_number[val_data[h2]])
        list_phe1.append(list_phe[val_data[h2]])
        dictSeq1[list_number1[h2]] = dictSeq[list_number1[h2]]
        if scheme != 'M1':
            dictSeq1_1[list_number1[h2]] = dictSeq_1[list_number1[h2]]

    mdata_train = myDataset(list_number2, list_phe2, dictSeq2, dictSeq2_1, scheme) if scheme != 'M1' \
                else myDataset(list_number2, list_phe2, dictSeq2, data_E, scheme)
    mdata_val = myDataset(list_number1, list_phe1, dictSeq1, dictSeq1_1, scheme) if scheme != 'M1' \
                else myDataset(list_number1, list_phe1, dictSeq1, data_E, scheme)

    return mdata_train, mdata_val


def train_Data(net, train_loader, optimizer, loss_func):
    net.train()
    train_runing_loss = 0.0
    for j, (id, phe, dictseq, data_x, data_stamp) in enumerate(train_loader, 0):
        dictseq = torch.stack(dictseq).to(args.device)
        phe = phe.clone().detach().to(args.device)
        phe = phe.float()
        dictseq = dictseq.float()
        data_x = data_x.to(args.device)
        data_stamp = data_stamp.to(args.device)

        optimizer.zero_grad()

        pred = net(dictseq, data_x, data_stamp).flatten()

        loss = loss_func(pred, phe)

        loss.backward()
        optimizer.step()

        train_runing_loss += loss.item()
    return train_runing_loss


def eval_Data(net, val_loader, loss_func, all_val_pred, all_val_phe):
    net.eval()
    val_runing_loss = 0.0
    for jj, (id, phe, dictseq, data_x, data_stamp) in enumerate(val_loader, 0):
        dictseq = torch.stack(dictseq).to(args.device)
        phe = phe.clone().detach().to(args.device)
        phe = phe.float()
        dictseq = dictseq.float()
        data_x = data_x.to(args.device)
        data_stamp = data_stamp.to(args.device)

        pred = net(dictseq, data_x, data_stamp).flatten()

        loss = loss_func(pred, phe)
        val_runing_loss += loss.item()

        phe = phe.cpu().detach().numpy().tolist()
        all_val_phe.extend(phe)
        pred = pred.flatten().cpu().detach().numpy().tolist()
        all_val_pred.extend(pred)
    return val_runing_loss


### Adjustment Parameter
def objective(trial):
    mdata_train, mdata_val=split_data(args.num_fold)

    batch = trial.suggest_int("batch", args.batch_1, args.batch_2)

    train_loader = DataLoader(
        dataset=mdata_train,
        batch_size=batch,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=mdata_val,
        batch_size=batch,
        shuffle=True,
    )

    net = GEFormer(args, snp_len, env_days, trial).to(args.device)
    lr = trial.suggest_float("lr", args.lr_1, args.lr_2, log=True)
    optimizer = optim.Adam(net.parameters(), lr)
    loss_func = nn.MSELoss()
    best_acc = 0.0
    for epoch in range(100):

        train_Data(net, train_loader, optimizer, loss_func)

        all_val_pred = []
        all_val_phe = []
        eval_Data(net, val_loader, loss_func,all_val_pred, all_val_phe)

        all_val_pred = np.asarray(all_val_pred)
        all_val_phe = np.asarray(all_val_phe)

        pccs = pearsonr(all_val_pred, all_val_phe)
        if pccs[0] > best_acc:
            best_acc = pccs[0]

    return best_acc


if args.need_optuna:
    path = './result/'+scheme+'/' + args.env_name+'/' + args.pheno_name
    if not os.path.exists(path):
        os.makedirs(path)
    storage_name = "sqlite:///./result/{}/{}/{}/optuna.db".format(scheme, args.env_name, args.pheno_name)
    studyname = str(time.time())
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20), direction="maximize",
        study_name=studyname, storage=storage_name, load_if_exists=True
    )
    study.optimize(objective, args.optuna_epoch)
    best_params = study.best_params
    best_value = study.best_value


    args.batch = best_params['batch']
    args.depth = best_params['depth']
    args.dropout = best_params['dropout']
    args.lr = best_params['lr']
    args.neurons1 = best_params['neurons1']
    args.neurons2 = best_params['neurons2']

### Start train
best_accz = float('-inf')

Kbest = []

k = [n + 1 for n in range(env_num)] if scheme == 'M2' else [n + 1 for n in range(10)]

for i in k:
    best_model = {}
    best_model_0 = {}

    mdata_train, mdata_val = split_data(i)

    train_loader = DataLoader(
        dataset=mdata_train,
        batch_size=args.batch,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=mdata_val,
        batch_size=args.batch,
        shuffle=True,
    )

    net = GEFormer(args, snp_len, env_days).to(args.device)
    optimizer = optim.Adam(net.parameters(), args.lr)
    loss_func = nn.MSELoss()
    best_acc = 0.0
    p_value = 0.0
    best_acc_0 = 0.0
    p_value_0 = 0.0
    for epoch in range(100):
        train_loss = []
        val_loss = []

        train_runing_loss = train_Data(net, train_loader, optimizer, loss_func)
        train_loss.append(train_runing_loss)
        # print("train_loss", int(train_loss[0]) / len(mdata_train))


        all_val_pred = []
        all_val_phe = []
        val_runing_loss = eval_Data(net, val_loader, loss_func, all_val_pred, all_val_phe)
        val_loss.append(val_runing_loss)
        # print("val_loss", int(val_loss[0]) / len(mdata_val))

        all_val_pred = np.asarray(all_val_pred)
        all_val_phe = np.asarray(all_val_phe)
        pccs = pearsonr(all_val_pred, all_val_phe)
        if pccs[0] > best_acc:
            best_acc = pccs[0]
            p_value = pccs[1]
            best_model = copy.deepcopy(net)
            if best_acc > best_accz:
                best_accz = best_acc
                folder_path = './best_model/'+scheme
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                path = folder_path+'/model_'+args.pheno_name+'_'+args.env_name+'.pkl' if scheme != 'M2' \
                    else folder_path+'/model_'+args.pheno_name+'.pkl'
                torch.save(net, path)  # The best model among all the folds
        elif pccs[0] < best_acc_0:
            best_acc_0 = pccs[0]
            p_value_0 = pccs[1]
            best_model_0 = copy.deepcopy(net)

    folder_path = './k_model/'+scheme+'/'+args.pheno_name+'_'+args.env_name if scheme != 'M2' \
        else './k_model/'+scheme+'/'+args.pheno_name
    path1 = folder_path + '/model_'+str(i)+'.pkl'  # The best model in every fold
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if best_acc == 0.0:
        Kbest.append(best_acc_0)
        pearson = (best_acc_0, p_value_0)
        torch.save(best_model_0, path1)
    else:
        Kbest.append(best_acc)
        pearson = (best_acc, p_value)
        torch.save(best_model, path1)

    if scheme == 'M2':
        print("Pearson-{} = ({:.2f}, {:.4f})".format(i, pearson[0], pearson[1]))

if scheme != 'M2':
    print("Pearson:", sum(Kbest) / len(k))
