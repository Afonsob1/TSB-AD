# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-M/')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/multi/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/multi/')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--AD_Name', type=str, default='IForest')
    args = parser.parse_args()


    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True)

    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)

    write_csv = []

    filename = args.filename

    print('Processing:{} by {}'.format(filename, args.AD_Name))

    file_path = os.path.join(args.dataset_dir, filename)
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)

    # min max normalization per channel
    # data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

    label = df['Label'].astype(int).to_numpy()
    # print('data: ', data.shape)
    # print('label: ', label.shape)

    feats = data.shape[1]
    slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
    train_index = filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    print('data_train: ', data_train.shape)

    start_time = time.time()

    if args.AD_Name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
    elif args.AD_Name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
    else:
        raise Exception(f"{args.AD_Name} is not defined")

    end_time = time.time()
    run_time = end_time - start_time

    if isinstance(output, np.ndarray):
        print(f'Success at {filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
        np.save(target_dir+'/'+filename.split('.')[0]+'.npy', output)
    else:
        print(f'ERROR: At {filename}: '+ output)

    print('arg save: ', args.save)
    ### whether to save the evaluation result
    if args.save:
        print('Evaluation result: ')

        print(output.shape)
        print(output)

        print(label.shape)
        
        evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
        print('evaluation_result: ', evaluation_result)
        list_w = list(evaluation_result.values())
        
        list_w.insert(0, run_time)
        list_w.insert(0, filename)
        write_csv.append(list_w)

        ## Temp Save
        col_w = list(evaluation_result.keys())
        col_w.insert(0, 'Time')
        col_w.insert(0, 'file')
        w_csv = pd.DataFrame(write_csv, columns=col_w)
        print(f'{args.save_dir}{args.AD_Name}/{args.filename}')
        
        save_path = f'{args.save_dir}{args.AD_Name}/'
        os.makedirs(save_path, exist_ok = True)

        w_csv.to_csv(f'{save_path}/{args.filename}', index=False)
        print('CSV saved successfully!')
        
    else:
        print('No evaluation result to save.')