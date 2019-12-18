import os

import numpy as np
import pandas as pd
import json
import copy

import torch
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True
torch.manual_seed(8) # for reproduce

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from AttentiveFP import (
    Fingerprint, 
    Fingerprint_viz, 
    save_smiles_dicts, 
    get_smiles_dicts, 
    get_smiles_array, 
    moltosvg_highlight)
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    precision_recall_curve,
    auc,
    f1_score)
from rdkit import Chem
import pickle

from bokeh.palettes import Category20_20, Category20b_20

def dim_reduce_op(values, op_type = 'seq'):
    if op_type == 'seq':
        pca = PCA(n_components = 20)
        value_reduced_20d = pca.fit_transform(values)
        tsne = TSNE(n_components = 2, perlexity=30, n_iter=1000)
        value_reduced = tsne.fit_transform(value_reduced_20d)
    return value_reduced

def cluster_MiniBatch(values, grain_size = 30):
    # clustering using KMeans minibatch algorithm
    n_clusters = int(values.shape[0] / grain_size)
    n_clusters = min(n_clusters, 500)
    mbk = MiniBatchKMeans(init = 'k-means++', init_size = 501, n_clusters = n_clusters, batch_size=100,
                      n_init = 10, max_no_improvement=10, verbose=0, random_state=0)
    mbk.fit(values)
    return mbk

def update_bicluster(batch_df, task_df, compound_df, mode = 'RobustMT', K = 5):
    if mode == 'RobustMT':
        n_tasks = task_df.shape[1] - 1
    elif mode == 'ST':
        n_tasks = 1
    elif mode == 'MT':
        n_tasks = task_df.shape[1]
    
    if not mode == 'ST':
        # cocluster of the minibatch predictive matrix
        X = preprocessing.scale(np.matrix(batch_df)[:,0:n_tasks])
        cocluster = SpectralCoclustering(n_clusters=K, random_state=0)
        cocluster.fit(X)
        batch_df['batch_label'] = cocluster.row_labels_
    else:
        rank_x = batch_df[batch_df.columns[0]].rank().tolist()
        groups = pd.qcut(rank_x, K, duplicates='drop')
        batch_df['batch_label'] = groups.codes
    
    # generate color hex for batch_label
    lut = dict(zip(batch_df['batch_label'].unique(), Category20_20))
    batch_df['batch_label_color'] = batch_df['batch_label'].map(lut)
    
    # generate color hex for compound_df
    lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
    compound_df['batch_label_color'] = compound_df['label'].map(lut2)
    lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
    compound_df['batch_label'] = compound_df['label'].map(lut22)
    groups = pd.qcut(compound_df['label'].tolist(), len(Category20b_20), duplicates='drop')
    c = [Category20b_20[xx] for xx in groups.codes]
    compound_df['label_color'] = c
    
    return batch_df, task_df, compound_df


def data_prepare_AttentiveFP(task_name, raw_filename, smiles_field,
                            cano_field = 'cano_smiles'):
    '''
    INPUT
        task_name: user-defined name for the training project
        raw_filename: a csv file containing smiles and task values of compounds
    '''
    feature_filename = raw_filename.replace('.csv','.pickle')
    filename = raw_filename.replace('.csv','')
    prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
    output_filename = filename + '_processed.csv'
    
    print('============== Loading the raw file =====================')
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df[smiles_field].values
    print("number of all smiles: ", len(smilesList))
    
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print('not successfully processed smiles: ', smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df[smiles_field].isin(remained_smiles)]
    smiles_tasks_df[cano_field] = canonical_smiles_list
    assert canonical_smiles_list[8] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df[cano_field][8]), 
                                                        isomericSmiles=True)
    smiles_tasks_df.to_csv(output_filename, index = None)
    print('saving processed file as ' + output_filename)

    print('================== saving feature files ========================')
    smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms()) < 151]
    if os.path.isfile(feature_filename):
        print('feature file has been generated.')
    else:
        feature_dicts = save_smiles_dicts(smilesList, feature_filename)
        print('saving feature file as ', feature_filename)
    return atom_num_dist



def train_regressor(model, dataset, tasks, optimizer, loss_function, 
                    batch_size, smiles_field, normalizeFlag, feature_dicts, stats):
    ratio_list = stats['ratio'].values
    model.train()
    #np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])
    # shuffle dataset
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df[smiles_field].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = \
                                                get_smiles_array(smiles_list, feature_dicts)
        atoms_prediction, mol_prediction, _, _ = model(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                                       torch.cuda.LongTensor(x_atom_index),
                                                       torch.cuda.LongTensor(x_bond_index),
                                                       torch.Tensor(x_mask))
        optimizer.zero_grad()
        loss = 0.0
        # compute the loss function
        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i]
            y_val = batch_df[task+normalizeFlag].values  # need to check how normalization deal with NAs
            # filter out NAs
            validInds = np.where(~np.isnan(y_val))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss += loss_function(
                y_pred_adjust,
                torch.Tensor(y_val_adjust).squeeze())*ratio_list[i]**2
        loss.backward()
        optimizer.step()
    return loss


def eval_regressor(model, dataset, smiles_field, tasks, normalizeFlag, 
                   batch_size, feature_dicts, stats, plot_flag = False):
    std_list = stats['Standard deviation'].values
    ratio_list = stats['ratio'].values
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    eval_MAE_list = {}
    eval_MSE_list = {}

    for i, task in enumerate(tasks):
        y_pred_list[task] = np.array([])
        y_val_list[task] = np.array([])
        eval_MAE_list[task] = np.array([])
        eval_MSE_list[task] = np.array([])

    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:(i+batch_size)]
        batch_list.append(batch)
    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch, :]
        smiles_list = batch_df[smiles_field].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list, feature_dicts)
        atoms_prediction, mol_prediction, _, _ = model(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                                       torch.cuda.LongTensor(x_atom_index),
                                                       torch.cuda.LongTensor(x_bond_index),
                                                       torch.Tensor(x_mask))

        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i]
            y_val = batch_df[task+normalizeFlag].values
            # filter out NAs
            validInds = np.where(~np.isnan(y_val))[0]
            valid_len = len(validInds)
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            MAE = F.l1_loss(y_pred_adjust,
                            torch.Tensor(y_val_adjust).squeeze(), reduction = 'none')
            MSE = F.mse_loss(y_pred_adjust,
                             torch.Tensor(y_val_adjust).squeeze(), reduction = 'none')
            y_pred_list[task] = np.concatenate([y_pred_list[task],  y_pred_adjust.cpu().detach().numpy()])
            y_val_list[task] = np.concatenate([y_val_list[task], y_val_adjust])
            
            if valid_len == 1:
                eval_MAE_list[task] = np.concatenate([eval_MAE_list[task], MAE.data.squeeze().cpu().numpy().reshape((1,))])
                eval_MSE_list[task] = np.concatenate([eval_MSE_list[task], MSE.data.squeeze().cpu().numpy().reshape((1,))])
            else:
                eval_MAE_list[task] = np.concatenate([eval_MAE_list[task], MAE.data.squeeze().cpu().numpy()])
                eval_MSE_list[task] = np.concatenate([eval_MSE_list[task], MSE.data.squeeze().cpu().numpy()])
    
    eval_r2_score = np.array([r2_score(y_val_list[task], y_pred_list[task]) for task in tasks])
    eval_MSE_normalized = np.array([eval_MSE_list[task].mean() for i, task in enumerate(tasks)])
    eval_MAE_normalized = np.array([eval_MAE_list[task].mean() for i, task in enumerate(tasks)])
    eval_MAE = np.multiply(eval_MAE_normalized, np.array(std_list))
    eval_MSE = np.multiply(eval_MSE_normalized, np.array(std_list))

    if plot_flag:
        return eval_r2_score, eval_MAE_normalized, eval_MAE, eval_MSE_normalized, eval_MSE, y_pred_list, y_val_list

    return eval_r2_score, eval_MAE_normalized, eval_MAE, eval_MSE_normalized, eval_MSE



def AttentiveFP_regressor_training(df_filename, feature_filename, tasks,
                                fingerprint_dim, radius, T, output_dir,
                                smiles_field = 'cano_smiles', normalizeFlag = '_normalized',
                                test_fraction = 10, random_seed = 8,
                                batch_size = 128, epochs = 300, p_dropout = 0.5,
                                weight_decay = 4.9, learning_rate = 3.4,
                                batch_normalization = False):
    '''
    INPUT:
    df - a dataframe recording values for tasks;
    feature_filename - .p file name of the stored chemical feature dictionary;
    tasks - a list, must be a subset of df.columns;
    fingerprint_dim - the number of nodes in hidden layer;
    radius - the number of recurrent layers on molecular graph;
    T - the number of recurrent layers on virtual graph;
    '''

    #1 prepare dataset (just extract needed subset, id + targets + smiles), split for train/test
    print('============ Training data loading =================')
    df = pd.read_csv(df_filename)
    feature_dicts = pickle.load(open(feature_filename, 'rb'))
    remained_df = df[df[smiles_field].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    uncovered_df = df.drop(remained_df.index)
    if len(uncovered_df) > 0:
        print('The following data is missing:')
        print(uncovered_df)

    test_df = remained_df.sample(frac = 1/test_fraction, random_state = random_seed)
    training_data = remained_df.drop(test_df.index)
      # get the stats of the training data, which will be used to normalize the loss
    columns = ['Task', 'Mean', 'Standard deviation', 'Mean absolute deviation', 'ratio']
    mean_list = []
    std_list = []
    mad_list = []
    ratio_list = []
    
    for task in tasks:
        mean = training_data[task].mean()
        mean_list.append(mean)
        std = training_data[task].std()
        std_list.append(std)
        mad = training_data[task].mad()
        mad_list.append(mad)
        ratio_list.append(std/mad)
        training_data[task+normalizeFlag] = (training_data[task] - mean) / std
        test_df[task+normalizeFlag] = (test_df[task] - mean) / std

    list_of_tuples = list(zip(tasks, mean_list, std_list, mad_list, ratio_list))
    stats = pd.DataFrame(list_of_tuples, columns = columns)
    stats.to_csv(output_dir + 'trainset_stats.csv', index = None)

    train_df = training_data.reset_index(drop = True)
    test_df = test_df.reset_index(drop=True)
    print('Data loading finished:')
    print('Train set size: %i' % len(train_df))
    print('Test set size: %i' % len(test_df))

    #2 model initialization
    print('============ Model initialization =================')
    per_task_output_units_num = 1
    output_units_num = len(tasks) * per_task_output_units_num

    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = \
                    get_smiles_array([remained_df[smiles_field].iloc[0]], feature_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]
    loss_function = nn.MSELoss()

    model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                        fingerprint_dim, output_units_num, p_dropout, batch_normalization)
    model.cuda()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total number of parameters: %i' % params)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    optimizer = optim.Adam(model.parameters(), 10**-learning_rate, 
                           weight_decay=10**-weight_decay)

    print('============ Saving params =================')
    #5 write all params to json file
    model_params = {}
    model_params['radius'] = radius
    model_params['fingerprint_dim'] = fingerprint_dim
    model_params['T'] = T
    model_params['output_units_num'] = output_units_num
    model_params['num_atom_features'] = num_atom_features
    model_params['num_bond_features'] = num_bond_features
    model_params['p_dropout'] = p_dropout
    model_params['batch_normalization'] = batch_normalization
    model_params['mol_length'] = x_atom.shape[1]

    data_stats = {}
    data_stats['tasks'] = tasks
    data_stats['smiles_field'] = smiles_field
    data_stats['test_fraction'] = test_fraction
    data_stats['random_seed'] = random_seed
    data_stats['mean'] = mean_list
    data_stats['std'] = std_list
    data_stats['mad'] = mad_list
    data_stats['ratio'] = ratio_list

    training_params = {}
    training_params['batch_size'] = batch_size
    training_params['epochs'] = epochs
    training_params['weight_decay'] = weight_decay
    training_params['learning_rate'] = learning_rate
    training_params['normalizeFlag'] = normalizeFlag

    json_output = {}
    json_output['model_params'] = model_params
    json_output['data_stats'] = data_stats
    json_output['training_params'] = training_params

    with open(output_dir + 'params.json', 'w') as outfile:
        json.dump(json_output, outfile)


    #3 model training
    print('============ Start model training =================')
      # parameter initialization
    for m in model.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, (nn.GRUCell)):
            nn.init.orthogonal_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)

    best_param ={}
    best_param["train_epoch"] = 0
    best_param["valid_epoch"] = 0
    best_param["train_MSE_normalized"] = 9e8
    best_param["valid_MSE_normalized"] = 9e8

    for epoch in range(epochs):
        print(train_regressor(model, train_df, tasks, optimizer, loss_function, 
                              batch_size, smiles_field, normalizeFlag, feature_dicts, stats))
        train_r2, train_MSE_normalized, train_MSE, train_MAE_normalized, \
                    train_MAE = eval_regressor(model, train_df, smiles_field, tasks, normalizeFlag, \
                                               batch_size, feature_dicts, stats)
        valid_r2, valid_MSE_normalized, valid_MSE, valid_MAE_normalized, \
                    valid_MAE = eval_regressor(model, test_df, smiles_field, tasks, normalizeFlag, \
                                               batch_size, feature_dicts, stats)

    #4 evluation and log tracking
        print("EPOCH:\t" + str(epoch) + '\n' \
            +"train_MAE: \n" + str(train_MAE) + '\n' \
            +"valid_MAE: \n" + str(valid_MAE) + '\n' \
            +"train_r2: \n" + str(train_r2) + '\n' \
            +"valid_r2: \n" + str(valid_r2) + '\n' \
            +"train_MSE_normalized_mean: " + str(train_MSE_normalized.mean()) + '\n' \
            +"valid_MSE_normalized_mean: " + str(valid_MSE_normalized.mean()) + '\n' \
            +"train_r2_mean: " + str(train_r2.mean()) + '\n' \
            +"valid_r2_mean: " + str(valid_r2.mean()) + '\n')
        if train_MSE_normalized.mean() < best_param["train_MSE_normalized"]:
            best_param["train_epoch"] = epoch
            best_param["train_MSE_normalized"] = train_MSE_normalized.mean()
        if valid_MSE_normalized.mean() < best_param["valid_MSE_normalized"]:
            best_param["valid_epoch"] = epoch
            best_param["valid_MSE_normalized"] = valid_MSE_normalized.mean()
            if valid_r2.mean() > 0.6:
                torch.save(model, output_dir + 'model-' + str(epoch) + '.pt')
        if (epoch - best_param["train_epoch"] > 3) and (epoch - best_param["valid_epoch"] > 5): # early stopping
            torch.save(model, output_dir + 'model-' + str(epoch) + '.pt')
            break
    print("Training finished.")

    return 


def generate_RUNKEY_dataframe_AttentiveFP(df_filename, feature_filename, param_file_name,
                                          prev_model, id_field, layer = 0, batch_size = 128):
    # load parameters
    print('-------------- Load parameters --------------')
    with open(param_file_name, 'r') as myfile:
        data = myfile.read()
    obj = json.loads(data)

    normalizeFlag = obj['training_params']['normalizeFlag']
    smiles_field = obj['data_stats']['smiles_field']
    tasks = obj['data_stats']['tasks']

    # load total datasets
    print('-------------- Load dataset --------------')
    df = pd.read_csv(df_filename)
    feature_dicts = pickle.load(open(feature_filename, 'rb'))
    remained_df = df[df[smiles_field].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    uncovered_df = df.drop(remained_df.index)
    if len(uncovered_df) > 0:
        print('The following data is missing:')
        print(uncovered_df)

    df = remained_df[tasks + [obj['data_stats']['smiles_field']] + [id_field]]
    if not type(df[id_field].iloc[0]) == str:
        df['chembl_id'] = df[id_field].astype(int)
    else:
        df['chembl_id'] = df[id_field]
    switch_field = lambda item:'canonical_smiles' if  item == smiles_field  else item
    df.columns = [switch_field(item) for item in df.columns.tolist()]
    
    
    # load previously trained model
    print('-------------- Load prev model --------------')
    best_model = torch.load(prev_model)
    best_model_dict = best_model.state_dict()
    best_model_wts = copy.deepcopy(best_model_dict)

    model_for_viz = Fingerprint_viz(obj['model_params']['radius'], obj['model_params']['T'],
                                    obj['model_params']['num_atom_features'], obj['model_params']['num_bond_features'],
                                    obj['model_params']['fingerprint_dim'], obj['model_params']['output_units_num'],
                                    obj['model_params']['p_dropout'], obj['model_params']['batch_normalization'])
    model_for_viz.load_state_dict(best_model_wts)

    # calculate prediction and coords (maybe need batch process)
    print('-------------- prepare compound df --------------')
    valList = np.arange(0, df.shape[0])
    np.random.shuffle(valList)
    df = df.loc[valList,:]
    df = df.reset_index(drop = True)

    N_training = min([df.shape[0], 5000])
    valList = np.arange(0, N_training)
    compound_df = df.loc[valList,:]
    batch_list = []
    for i in range(0, N_training, batch_size):
        batch = valList[i:(i+batch_size)]
        batch_list.append(batch)
    
    pred_mat = np.zeros((N_training, len(tasks)))
    atom_weight_mat = np.zeros((N_training, obj['model_params']['mol_length']))
    mol_feature_mat = np.zeros((N_training, obj['model_params']['fingerprint_dim']))

    for counter, train_batch in enumerate(batch_list):
        temp_df = compound_df.loc[train_batch,:]
        smiles_list = temp_df['canonical_smiles'].tolist()
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, \
            mol_attention_weight_viz, mol_prediction = model_for_viz(
                torch.Tensor(x_atom), torch.Tensor(x_bonds),
                torch.cuda.LongTensor(x_atom_index),
                torch.cuda.LongTensor(x_bond_index),
                torch.Tensor(x_mask))

        mol_pred = np.array(mol_prediction.data.squeeze().cpu().numpy())
        mol_pred_translate = np.zeros(mol_pred.shape)
        for ii, task in enumerate(tasks):
            pos = int(np.where(np.array(obj['data_stats']['tasks'])==task)[0])
            mol_pred_translate[:,ii] = (mol_pred[:,pos] * obj['data_stats']['std'][pos]) + obj['data_stats']['mean'][pos]
        pred_mat[train_batch,:] = mol_pred_translate
    
        atom_weight = np.stack([mol_attention_weight_viz[t].cpu().detach().numpy() for t in range(obj['model_params']['T'])])[layer,:,:,0]
        # arrange the atoms according to its index
        for m in range(len(temp_df)):
            smiles = smiles_list[m]
            ind_mask = x_mask[m]
            ind_atom = smiles_to_rdkit_list[smiles]
            ind_weight = atom_weight[m,:]

            out_weight = []
            for j, one_or_zero in enumerate(list(ind_mask)):
                if one_or_zero == 1.0:
                    out_weight.append(ind_weight[j])
            out_weight_sorted = np.array([out_weight[m] for m in np.argsort(ind_atom)]).flatten()
            atom_weight_mat[train_batch[m], 0:len(out_weight_sorted)] = out_weight_sorted

        mol_feature_mat[train_batch,:] = np.stack([mol_feature_viz[t].cpu().detach().numpy() for t in range(obj['model_params']['T'])])[layer,:,:]
    
    # prepare compound_df (dim reduce + minibatch clustering)
    coord_values = dim_reduce_op(transfer_values, type = 'seq')
    compound_df['x'] = coord_values[:,0]
    compound_df['y'] = coord_values[:,1]
    for ii, task in enumerate(tasks):
        compound_df['pred_' + task] = pred_mat[:, ii]

    mbk = cluster_MiniBatch(coord_values)
    mbk.means_labels_unique = np.unique(mbk.labels_)
    compound_df['label'] = mbk.labels_

    # prepare batch_df
    print('-------------- prepare batch df --------------')
    n_row = len(mbk.means_labels_unique)
    n_col = len(tasks)
    cluster_info_mat = np.zeros((n_row, (n_col + 3)))

    for k in range(n_row):
        mask = mbk.labels_ == mbk.means_labels_unique[k]
        cluster_info_mat[k, 0:n_col] = np.nanmean(pred_mat[mask], axis = 0)
        cluster_info_mat[k, n_col] = sum(mask)
        cluster_info_mat[k, (n_col + 1) : (n_col + 3)] = np.nanmean(coord_values[mask, :], axis = 0)
    batch_df = pd.DataFrame(cluster_info_mat)
    batch_df.columns = ['avg_' + task for task in tasks] + ['size', 'coordx', 'coordy']
    batch_df['Label_id'] = mbk.means_labels_unique

    # prepare task_df
    print('-------------- prepare task df --------------')
    task_df = pd.DataFrame(atom_weight_mat)

    ### generate color labels by default
    print('------- Generate color labels with default K of 5 --------')
    batch_df, task_df, compound_df = update_bicluster(batch_df, task_df, compound_df, mode = 'ST')
    
    ### wrapping up
    print('-------------- Saving datasets ----------------')
    compound_df.to_csv(output_prefix + 'compound_df.csv', index = False)
    batch_df.to_csv(output_prefix + 'batch_df.csv', index = False)
    task_df.to_csv(output_prefix + 'task_df.csv', index = False)

    return
