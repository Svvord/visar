# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:07:01 2019

@author: Dingqy
"""
import os

import numpy as np
import deepchem as dc
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from keras.layers import Dense, Input
from keras.layers.core import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn import linear_model
from sklearn.svm import LinearSVR

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import preprocessing


#---------------------------------------------
# baseline model training
def optimize_SVR(X_train, Y_train):
    # input: X_train --- each row a sample; each col a feature
    #        Y_train 
    # output: a predictive model of support vector regression
    clf = LinearSVR(C=1.0, epsilon=0.2)
    try:
        clf.fit(X_train, Y_train)
    except Exception as e:
        print(e)
    return clf

def optimize_RidgeCV(X_train, Y_train):
    '''
    input: X_train --- each row a sample; each col a feature
           Y_train 
    output: a predictive model of ridge regression with selected alpha by cross-validation
    '''
    alphas = np.logspace(start = -1, stop = 2, num = 20)
    reg = linear_model.RidgeCV(alphas)
    try:
        reg.fit(X_train, Y_train)
    except Exception as e:
        print(e)
    return reg

#---------------------------------------------
def prepare_dataset(fname, task, dataset_file, FP_type, 
                    model_flag = 'ST', add_features = None,
                    smiles_field = 'salt_removed_smi', id_field = 'molregno'):
    '''
    input: fname --- name of the file of raw data containing chemicals and the value for each assay;
           task --- list of task names (supporting 1 or more, but must be a list)
           dataset_file --- name of the temperary file saving intermediate dataset containing only the data for a specific training task
           FP_type --- name of the fingprint type (one of 'Circular_2018', 'Circular_1024', 'Morgan', 'MACCS', 'RDKit_FP')
           model_flag --- type of the model (ST: single task, len(task)==1; MT: multitask, len(task)>=1)
           add_features --- must be None for ST; list of properties
           smiles_field --- the name of the field in fname defining the SMILES for chemicals
           id_field --- the name of the field in fname defining the identifier for chemicals
    output: dataset build by deepchem
    '''
    MT_df = pd.read_csv(fname)
    if model_flag == 'ST':
        df = extract_clean_dataset(task, MT_df, smiles_field = smiles_field, id_field = id_field)
    elif model_flag == 'MT':
        df = extract_clean_dataset(task, MT_df, add_features = ['MW','logP','TPSA','BertzCT'], smiles_field = smiles_field, id_field = id_field)
        task = task + add_features
    
    if FP_type == 'Circular_2048':
        df.to_csv(dataset_file)
        featurizer = dc.feat.CircularFingerprint(size=2048)
        loader = dc.data.CSVLoader(id_field=id_field, 
                                   smiles_field=smiles_field, 
                                   tasks = task,
                                   featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
        
    elif FP_type == 'Circular_1024':
        df.to_csv(dataset_file)
        featurizer = dc.feat.CircularFingerprint(size=1024)
        loader = dc.data.CSVLoader(id_field=id_field, 
                                   smiles_field=smiles_field, 
                                   tasks = task,
                                   featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
    
    elif FP_type == 'RDKit_FP':
        df.to_csv(dataset_file)
        featurizer = dc.feat.RDKitDescriptors()
        loader = dc.data.CSVLoader(id_field=id_field, 
                                   smiles_field=smiles_field, 
                                   tasks = task,
                                   featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
        
    elif FP_type == 'Morgan': #2048
        smiles = df[smiles_field].tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        fps = []
        for mol in mols:
            fps.append(np.matrix(AllChem.GetMorganFingerprintAsBitVect(mol,2)))
        fps = np.concatenate(fps)
        new_df = pd.DataFrame(fps)
        new_df.columns = ['V' + str(i) for i in range(fps.shape[1])]
        new_df[id_field] = df[id_field]
        for m in range(len(task)):
            new_df[task[m]] = df[task[m]]
        new_df.to_csv(dataset_file)
        user_specified_features = ['V' + str(i) for i in range(fps.shape[1])]
        featurizer = dc.feat.UserDefinedFeaturizer(user_specified_features)
        loader = dc.data.UserCSVLoader(
                tasks=task, smiles_field=smiles_field, id_field=id_field,
                featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
        
    elif FP_type == 'MACCS': #167
        smiles = df[smiles_field].tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        fps = []
        for mol in mols:
            fps.append(np.matrix(MACCSkeys.GenMACCSKeys(mol)))
        fps = np.concatenate(fps)
        new_df = pd.DataFrame(fps)
        new_df.columns = ['V' + str(i) for i in range(fps.shape[1])]
        new_df[id_field] = df[id_field]
        for m in range(len(task)):
            new_df[task[m]] = df[task[m]]
        new_df.to_csv(dataset_file)
        user_specified_features = ['V' + str(i) for i in range(fps.shape[1])]
        featurizer = dc.feat.UserDefinedFeaturizer(user_specified_features)
        loader = dc.data.UserCSVLoader(
                tasks = task, smiles_field=smiles_field, id_field=id_field,
                featurizer=featurizer)
        dataset = loader.featurize(dataset_file)
    
    else:
        print('Unsupported Fingerprint type!')
    
    return dataset, df
#----------------------------------------------
def extract_clean_dataset(subset_names, MT_df, add_features = None, id_field = 'molregno', smiles_field = 'salt_removed_smi'):
    '''
    input: subset_names --- a list of the task names
           MT_df --- pd.DataFrame of the total raw data
    output: subset of raw data， containing only the selected tasks
    '''
    extract_column = subset_names + [id_field, smiles_field]
    sub_df = MT_df[extract_column]
    
    n_tasks = len(subset_names)
    mask_mat = np.zeros((len(sub_df), n_tasks))
    for j in range(n_tasks):
        mask_mat[:,j] = np.isnan(np.array(sub_df[subset_names[j]].tolist()))
    mask_new = np.sum(mask_mat, axis = 1) < n_tasks
    
    if not add_features:
        extract_df = sub_df.loc[mask_new]
    else:
        new_sub_df = MT_df[extract_column + add_features]
        # normalization!!
        for k in range(len(add_features)):
            new_sub_df.loc[:,add_features[k]] = preprocessing.scale(np.array(new_sub_df[add_features[k]].tolist()))
        extract_df = new_sub_df.loc[mask_new]
    print('Extracted dataset shape: ' + str(extract_df.shape))
    return extract_df

#----------------------------------------------
def model_builder(model_params, model_dir):
    model = dc.models.MultitaskRegressor(**model_params)
    return model

def ST_model(n_feature, layer_size, drop_prob,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5):
    training_dat = Input(shape = (n_feature,), dtype = 'float32')
    X = Dense(layer_size[0], activation = 'relu')(training_dat)
    for i in range(1,len(layer_size)):
        X = Dropout(drop_prob)(X)
        X = Dense(layer_size[i], activation = 'relu')(X) 
    model = Model(input = training_dat, output = X)
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    model.compile(loss = 'mean_squared_error', optimizer = optimizer)
    return model

def ST_model_hyperparam_screen(fname, task_names, FP_type, params_dict, log_path = './logs/'):
    '''
    hyperparameter screening using deepchem package
    input: fname --- name of the file of raw data containing chemicals and the value for each assay;
           task --- list of task names (supporting 1 or more, but must be a list)
           FP_type --- name of the fingprint type (one of 'Circular_2018', 'Circular_1024', 'Morgan', 'MACCS', 'RDKit_FP')
           params_dict --- dictionary containing the parameters along with the screening range
           log_path --- the directory saving the log file
    output: the log; log file saved in log_path
    '''
    log_output = []
    for task in task_names:
        print('----------------------------------------------')
        dataset_file = '%s/temp.csv' % (log_path)
        dataset = prepare_dataset(fname, [task], dataset_file, FP_type)
        for cnt in range(3):
            print('Preparing dataset for %s of rep %d...' % (task, cnt))
            splitter = dc.splits.RandomSplitter(dataset_file)
            train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
    
            print('Hyperprameter screening ...')
            metric = dc.metrics.Metric(dc.metrics.r2_score)
            optimizer = dc.hyper.HyperparamOpt(model_builder)
            best_dnn, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict,
                                                                      train_dataset,
                                                                      valid_dataset, [],
                                                                      metric)
            # get the layer size and dropout rate of all_results
            for (key, value) in all_results.items():
                log_output.append('rep%d\t%s\t%s\t%s' % (cnt, task, str(key), str(value)))
    
            print('Generate performace report ...')
            with open('%s/hyperparam_log.txt' % (log_path), 'w') as f:
                for line in log_output:
                    f.write("%s\n" % line)
        os.system('rm %s' % dataset_file)
    
    return  log_output

def ST_model_training(fname, FP_type, best_hyperparams, result_path, epoch_num = 45,
                      frac_train = 0.8, N_test = None):
    '''
    Training the specific groups of single task (or multitask with only one task and a few chemical properties),
           using 
    input: fname --- name of the file of raw data containing chemicals and the value for each assay;
           FP_type --- name of the fingprint type (one of 'Circular_2018', 'Circular_1024', 'Morgan', 'MACCS', 'RDKit_FP')
           best_hyperparams --- dictionary containing the parameters, with key the name of the task
           result_path --- the directory saving the trained models and merits file
    output: the log; log file saved in log_path
    '''
    task_names = list(best_hyperparams.keys())
    output_dat = np.zeros((len(task_names) * 3, 12))
    rep_labels = ['NA' for _ in range(len(task_names) * 3)]
    task_labels = ['NA' for _ in range(len(task_names) * 3)]
    nrow = 0
    for task in task_names:
        print('----------------------------------------------')
        dataset_file = '%s/temp.csv' % (result_path)
        dataset = prepare_dataset(fname, [task], dataset_file, FP_type)

        for cnt in range(3):
            print('Preparing dataset for %s of rep %d...' % (task, cnt))
            splitter = dc.splits.RandomSplitter(dataset_file)
            if not N_test:
                train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train = frac_train)
            else:
                frac = 1 - N_test / dataset.X.shape[0]
                train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train = frac)
            
            print('Model training ...')
            n_features = train_dataset.X.shape[1]
            model = ST_model(n_features, best_hyperparams[task][0], best_hyperparams[task][1], lr=0.0001)
            for iteration in range(51):
                checkpoint_callback = ModelCheckpoint('%s/%s_rep%d_%d.hdf5' % (result_path, task,
                                                                                cnt, iteration),
                                          monitor='val_loss', verbose = 0, save_weights_only = False,
                                          mode = 'auto', period = 1)
                if iteration % 10 == 0:
                    model.fit(train_dataset.X, train_dataset.y, nb_epoch=5, callbacks = [checkpoint_callback])
                else:
                    model.fit(train_dataset.X, train_dataset.y, nb_epoch=epoch_num, verbose=0)
            
            print('Training baseline models ...')
            X_new = np.c_[[1]*train_dataset.X.shape[0], train_dataset.X]
            clf = optimize_SVR(X_new, train_dataset.y)
            reg = optimize_RidgeCV(X_new, train_dataset.y)
            
            #print('Training TRACE models ...')
            # todo!
            
            print('Saving metrics ...')
            y_true = test_dataset.y.flatten()
            y_pred_ST = model.predict(test_dataset.X).flatten()
            y_pred_SVR = clf.predict(np.c_[[1]*test_dataset.X.shape[0],test_dataset.X]).flatten()
            y_pred_Rig = reg.predict(np.c_[[1]*test_dataset.X.shape[0],test_dataset.X]).flatten()
            
            output_dat[nrow, 0] = sqrt(mean_squared_error(y_pred_ST, y_true))
            output_dat[nrow, 1] = sqrt(mean_squared_error(y_pred_SVR, y_true))
            output_dat[nrow, 2] = sqrt(mean_squared_error(y_pred_Rig, y_true))
            output_dat[nrow, 3], _ = pearsonr(y_pred_ST, y_true)
            output_dat[nrow, 4], _ = pearsonr(y_pred_SVR, y_true)
            output_dat[nrow, 5], _ = pearsonr(y_pred_Rig, y_true)
            
            y_true = train_dataset.y.flatten()
            y_pred_ST = model.predict(train_dataset.X).flatten()
            y_pred_SVR = clf.predict(np.c_[[1]*train_dataset.X.shape[0],train_dataset.X]).flatten()
            y_pred_Rig = reg.predict(np.c_[[1]*train_dataset.X.shape[0],train_dataset.X]).flatten()
            
            output_dat[nrow, 6] = sqrt(mean_squared_error(y_pred_ST, y_true))
            output_dat[nrow, 7] = sqrt(mean_squared_error(y_pred_SVR, y_true))
            output_dat[nrow, 8] = sqrt(mean_squared_error(y_pred_Rig, y_true))
            output_dat[nrow, 9], _ = pearsonr(y_pred_ST, y_true)
            output_dat[nrow, 10], _ = pearsonr(y_pred_SVR, y_true)
            output_dat[nrow, 11], _ = pearsonr(y_pred_Rig, y_true)
            
            rep_labels[nrow] = cnt
            task_labels[nrow] = task
            nrow += 1
        os.system('rm %s' % dataset_file)
    
        print('Generate performace report ...')
        output_df = pd.DataFrame(output_dat)
        output_df.columns = ['test_rmse_ST', 'test_rmse_SVR', 'test_rmse_Rig',
                             'test_pearsonr_ST', 'test_pearsonr_SVR', 'test_pearsonr_Rig',
                             'train_rmse_ST', 'train_rmse_SVR', 'train_rmse_Rig',
                             'train_pearsonr_ST', 'train_pearsonr_SVR', 'train_pearsonr_Rig']
        output_df['rep_label'] = rep_labels
        output_df['task_label'] = task_labels
        os.system('rm %s' % dataset_file)
        output_df.to_csv('%s/performance_metrics.csv' % (result_path))
    return  output_df
