
import numpy as np
import sys
import os
import random
from utils.dataset_generator import generate_data,generate_numpy_data
from sklearn.metrics import roc_auc_score

sys.path.append("..")

from utils.data_loader import CustomizeDataLoader
from utils.dataset_generator import generate_data

dataset = "CIFAR10"
normal_class= 0
model = "LinearROBODSub"
train_X, train_y =  generate_data(normal_class, dataset= dataset, transductive = True, flatten =False, GCN = True)
input_dim = train_X.shape[1] 
Files already downloaded and verified
with open("results/%s/%s/transductive/%d/%s.txt" % (model, dataset,normal_class, model)) as openfile: #
    results = openfile.readlines()
savedir = "results/%s/%s/transductive/%d" % (model, dataset, normal_class)
result_dict = {}
hp_set = set()
for i,line in enumerate(results):
    line = line.strip()
    if line.startswith("hpname"):
        if line not in hp_set:
            hp_set.add(line)
            result_dict[line] = {"time": [], "auroc": [], "memory": []}
    if line.startswith("exp_num"):
        hpname = results[i-1].strip()
        training_time = float(results[i+1].strip().split(": ")[1])
        auc = float(results[i+2].strip().split(": ")[1])
        memory = float(results[i+3].strip().split(": ")[1])
        result_dict[hpname]["time"].append(training_time)
        result_dict[hpname]["auroc"].append(auc)
        result_dict[hpname]["memory"].append(memory)
    else:
        continue
total_pred = []
total_time = []
total_memory =[]
for exp in range(3):
    pred = []
    time = 0.0
    max_memory = []
    for i in result_dict.keys():
        result = np.load(savedir + "/" +i.split("hpname: ")[1] + "/" + str(exp) + "_prediction.npy")
        individual_time = result_dict[i]['time'][exp]
        pred.append(result)
        time+= individual_time
        max_memory.append(result_dict[i]['memory'][exp])
    hyper_score = np.mean(pred, axis = 0)
    total_pred.append(roc_auc_score(train_y, hyper_score))
    total_time.append(time)
    total_memory.append(np.max(max_memory))
print("%.5f   %.5f" % (np.mean(total_pred), np.std(total_pred)))
0.59605   0.00522
print("%.5f   %.5f" % (np.mean(total_time), np.std(total_time)))
3229.61864   46.25541
