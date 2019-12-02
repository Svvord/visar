import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

from visar.model_training_utils import prepare_dataset
from visar.model_landscape_utils import (
    calculate_transfer_values,
    calculate_transfer_values_ST,
    dim_reduce,
    cluster_MiniBatch)
from visar.model_SAR_utils import calculate_gradients_RobustMT, calculate_gradients_ST

from scipy.stats import pearsonr
from bokeh.palettes import Category20_20, Category20b_20

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from sklearn import linear_model
from sklearn.svm import LinearSVR

import pdb

""" contribution from Hans de Winter """
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover

#----------------------------------------------------
def _InitialiseNeutralisationReactions():
    patts= (
        # Imidazoles
        ('[n+;H]','n'),
        # Amines
        ('[N+;!H0]','N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]','O'),
        # Thiols
        ('[S-;X1]','S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]','N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]','N'),
        # Tetrazoles
        ('[n-]','[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]','S'),
        # Amides
        ('[$([N-]C=O)]','N'),
        )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

_reactions=None
def NeutraliseCharges_RemoveSalt(smiles, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        remover = SaltRemover()
        mol, deleted = remover.StripMolWithDeleted(mol)
        replaced = False
        for i,(reactant, product) in enumerate(reactions):
            while mol.HasSubstructMatch(reactant):
                replaced = True
                rms = AllChem.ReplaceSubstructs(mol, reactant, product)
                mol = rms[0]
        if replaced:
            return (Chem.MolToSmiles(mol,True), True)
        else:
            return (smiles, False)
    else:
        return (None, False)
#--------------------------------------------------------

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


def generate_RUNKEY_dataframe_ST(prev_model, output_prefix, task_list, dataset_file, FP_type,
                                 add_features, mode = 'ST',
                                 MT_dat_name = './data/MT_data_clean_June28.csv', n_layer = 2,
                                 smiles_field = 'canonical_smiles', id_field = 'chembl_id',
                                 custom_file = None, custom_id_field = None, custom_task_field = None,
                                 custom_smiles_field = None, sep_custom_file = ','):
    if add_features is None:
        tasks = task_list
    else:
        tasks = task_list + add_features

    # load dataset
    print('------------- Loading dataset --------------------')
    dataset, df = prepare_dataset(MT_dat_name, task_list, dataset_file, FP_type,
                                  smiles_field = smiles_field,
                                  add_features = add_features,
                                  id_field = id_field, model_flag = 'ST')
    df['chembl_id'] = df[id_field].astype(int)
    switch_field = lambda item:'canonical_smiles' if  item == smiles_field  else item
    df.columns = [switch_field(item) for item in df.columns.tolist()]
    N_training = df.shape[0]

    # if custom file exisit, processing the file
    custom_file_flag = False
    if custom_file is not None:
        custom_file_flag = True
        print('------------- Loading custom file --------------------')
        custom_compound_df = pd.read_csv(custom_file, sep = sep_custom_file)

        ## checking point: the ids for custom files must be unique
        if len(set(custom_compound_df[custom_id_field])) < custom_compound_df.shape[0]:
            print('Error: The ids for custom files must be unique')
            return

        all_mols = custom_compound_df[custom_smiles_field].tolist()
        new_mols = all_mols
        valid_filter = [True for k in range(len(new_mols))]

        for i in range(len(new_mols)):
            (molSmiles, neutralised) = NeutraliseCharges_RemoveSalt(all_mols[i])
            if molSmiles is not None:
                new_mols[i] = molSmiles
            else:
                valid_filter[i] = False
            if i % 100 == 0:
                print(i)

        custom_compound_df['canonical_smiles'] = new_mols
        N_raw = len(custom_compound_df)
        custom_compound_df = custom_compound_df.loc[valid_filter]
        N_clean = len(custom_compound_df)
        #custom_compound_df['chembl_id'] = custom_compound_df[custom_id_field].astype(int)
        custom_compound_df['chembl_id'] = custom_compound_df[custom_id_field]
        print('Read in %d compounds; %d valid compounds.' % (N_raw, N_clean))

        if FP_type == 'Circular_2048':
            custom_compound_df.to_csv(dataset_file)
            featurizer = dc.feat.CircularFingerprint(size=2048)
            loader = dc.data.CSVLoader(id_field=custom_id_field,
                                   smiles_field='canonical_smiles',
                                   tasks = [custom_task_field],
                                   featurizer=featurizer)
            custom_dataset = loader.featurize(dataset_file)
        N_custom = custom_compound_df.shape[0]

    # load prev_model
    print('------------- Loading previous trained models ------------------')
    model = load_model(prev_model)

    ### for chemical information
    print('------------- Prepare information for chemicals ------------------')
    if custom_file_flag:
        values = np.concatenate((dataset.X, custom_dataset.X), axis = 0)
    else:
        values = dataset.X

    # build transfer model
    model_transfer = calculate_transfer_values_ST(prev_model=prev_model, n_layer = n_layer)
    coord_values = dim_reduce(model_transfer,values)
    pred_mat = model.predict(dataset.X)

    pred_df = pd.DataFrame(pred_mat)
    pred_df.columns = ['pred_' + xx for xx in tasks]
    pred_df['chembl_id'] = dataset.ids

    coord_df = pd.DataFrame(coord_values[0:N_training,:])
    coord_df.columns = ['x', 'y']
    coord_df['chembl_id'] = dataset.ids

    pred_df['chembl_id'] = pred_df['chembl_id'].astype(int)
    coord_df['chembl_id'] = coord_df['chembl_id'].astype(int)
    compound_df = pd.merge(df, coord_df, on = 'chembl_id')
    compound_df = pd.merge(compound_df, pred_df, on = 'chembl_id')

    if custom_file_flag:
        pred_custom_mat = model.predict(custom_dataset.X)

        pred_df2 = pd.DataFrame(pred_custom_mat)
        pred_df2.columns = ['pred_' + xx for xx in tasks]
        pred_df2['chembl_id'] = custom_dataset.ids

        coord_df2 = pd.DataFrame(coord_values[N_training:coord_values.shape[0],:])
        coord_df2.columns = ['x', 'y']
        coord_df2['chembl_id'] = custom_dataset.ids

        if not type(custom_compound_df[custom_id_field].iloc[0]) == str:
            coord_df2['chembl_id'] = coord_df2['chembl_id'].astype(int)
            pred_df2['chembl_id'] = pred_df2['chembl_id'].astype(int)
        compound_df2 = pd.merge(custom_compound_df, coord_df2, on = 'chembl_id')
        compound_df2 = pd.merge(compound_df2, pred_df2, on = 'chembl_id')

    ### for cluster information
    print('------------- Prepare information for minibatches ------------------')
    if custom_file_flag:
        pred_mat_concat = np.concatenate((pred_mat, pred_custom_mat), axis = 0)
    else:
        pred_mat_concat = pred_mat

    mbk = cluster_MiniBatch(coord_values)
    mbk.means_labels_unique = np.unique(mbk.labels_)
    n_row = len(np.unique(mbk.labels_))
    n_col = pred_mat.shape[1]
    cluster_info_mat = np.zeros((n_row, (n_col + 3)))

    for k in range(n_row):
        mask = mbk.labels_ == mbk.means_labels_unique[k]
        cluster_info_mat[k, 0:n_col] = np.nanmean(pred_mat_concat[mask,:], axis = 0)
        cluster_info_mat[k, n_col] = sum(mask)
        cluster_info_mat[k, (n_col + 1) : (n_col + 3)] = np.nanmean(coord_values[mask,:], axis = 0)
    compound_df['label'] = mbk.labels_[0:N_training]
    batch_df = pd.DataFrame(cluster_info_mat)
    batch_df.columns = ['avg_' + xx for xx in tasks] + ['size', 'coordx', 'coordy']
    batch_df['Label_id'] = mbk.means_labels_unique

    if custom_file_flag:
        compound_df2['label'] = mbk.labels_[N_training: len(mbk.labels_)]

    ### for task information
    print('------------- Prepare information for tasks ------------------')

    grad_mat = calculate_gradients_ST(dataset.X, prev_model)
    task_df = pd.DataFrame(grad_mat)

    ### generate color labels by default
    print('------- Generate color labels with default K of 5 --------')
    batch_df, task_df, compound_df = update_bicluster(batch_df, task_df, compound_df, mode = mode)

    if custom_file_flag:
        lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
        lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
        lut222 = dict(zip(compound_df['label'], compound_df['label_color']))
        compound_df2['batch_label_color'] = compound_df2['label'].map(lut2)
        compound_df2['batch_label'] = compound_df2['label'].map(lut22)
        compound_df2['label_color'] = compound_df2['label'].map(lut222)

    ### wrapping up
    print('-------------- Saving datasets ----------------')
    compound_df.to_csv(output_prefix + 'compound_df.csv', index = False)
    batch_df.to_csv(output_prefix + 'batch_df.csv', index = False)
    task_df.to_csv(output_prefix + 'task_df.csv', index = False)

    if custom_file_flag:
        compound_df2.to_csv(output_prefix + 'compound_custom_df.csv', index = False)

    return

def generate_performance_plot_ST(performance_file):
    output_df = pd.read_csv(performance_file)
    output_df = output_df.drop(['Unnamed: 0'], axis = 1)

    multi_index = [(output_df['rep_label'].iloc[xx], output_df['task_label'].iloc[xx]) for xx in range(len(output_df))]
    output_df['multi_index'] = multi_index
    output_df = output_df.set_index('multi_index')
    output_df = output_df.drop(['rep_label', 'task_label'], axis = 1)
    output_df.columns.name = 'metrics'

    plot_df = pd.DataFrame(output_df.stack(), columns = ['value']).reset_index()
    rep_vector = []
    task_vector = []
    tt_vector = []
    perf_vector = []
    method_vector = []
    for i in range(len(plot_df)):
        rep_vector.append(plot_df['multi_index'].iloc[i][0])
        task_vector.append(plot_df['multi_index'].iloc[i][1])
        tt_vector.append(plot_df['metrics'].iloc[i].split('_')[0])
        perf_vector.append(plot_df['metrics'].iloc[i].split('_')[1])
        method_vector.append(plot_df['metrics'].iloc[i].split('_')[2])
    plot_df['rep'] = rep_vector
    plot_df['task'] = task_vector
    plot_df['tt'] = tt_vector
    plot_df['performance'] = perf_vector
    plot_df['method'] = method_vector

    return plot_df

def generate_performance_plot_RobustMT(train_file, test_file):
    # load data
    test_df = pd.read_csv(test_file, header = 1, index_col = 0)
    test_df.index.name = 'step'
    train_df = pd.read_csv(train_file, header = 1, index_col = 0)
    train_df.index.name = 'step'

    # perpare dataframe for plotting
    test_df.columns.name = 'tasks'
    plot_test = pd.DataFrame(test_df.stack(), columns = ['R2']).reset_index()
    plot_test['tt'] = 'test'

    train_df.columns.name = 'tasks'
    plot_train = pd.DataFrame(train_df.stack(), columns = ['R2']).reset_index()
    plot_train['tt'] = 'train'

    plot_df = pd.concat([plot_train, plot_test])

    return plot_df


def generate_RUNKEY_dataframe_RobustMT(prev_model, output_prefix, task_list, dataset_file, FP_type, add_features,
                              n_features, layer_sizes, bypass_layer_sizes, model_flag, n_bypass,
                              MT_dat_name = './data/MT_data_clean_June28.csv', model_test_log = None,
                              smiles_field = 'canonical_smiles', id_field = 'chembl_id',
                              bypass_dropouts = [.5], dropout = 0.5, learning_rate = 0.001, n_layer = 2,
                              custom_file = None, custom_id_field = None, custom_task_field = None,
                              custom_smiles_field = None, sep_custom_file = ',', K = 5, valid_cutoff = None):
    if add_features is None:
        tasks = task_list
    else:
        tasks = task_list + add_features

    if model_test_log is None:
        n_tasks = len(tasks)
        model_tasks = tasks
    else:
        test_log_df = pd.read_csv(model_test_log, header = 1, index_col = 0)
        model_tasks = test_log_df.columns.values
        n_tasks = len(model_tasks)

    if valid_cutoff is not None:
        final_merit = test_log_df.iloc[-1,].values
        valid_mask = final_merit > valid_cutoff
    else:
        valid_mask = np.array([True] * n_tasks)
    # load dataset
    print('------------- Loading dataset --------------------')
    dataset, df = prepare_dataset(MT_dat_name, task_list, dataset_file, FP_type,
                                  smiles_field = smiles_field,
                                  add_features = add_features,
                                  id_field = id_field, model_flag = model_flag)
    df['chembl_id'] = df[id_field].astype(int)
    switch_field = lambda item:'canonical_smiles' if  item == smiles_field  else item
    df.columns = [switch_field(item) for item in df.columns.tolist()]
    N_training = df.shape[0]

    # if custom file exisit, processing the file
    custom_file_flag = False
    if custom_file is not None:
        custom_file_flag = True
        print('------------- Loading custom file --------------------')
        custom_compound_df = pd.read_csv(custom_file, sep = sep_custom_file)

        ## checking point: the ids for custom files must be unique
        if len(set(custom_compound_df[custom_id_field])) < custom_compound_df.shape[0]:
            print('Error: The ids for custom files must be unique')
            return

        all_mols = custom_compound_df[custom_smiles_field].tolist()
        new_mols = all_mols
        valid_filter = [True for k in range(len(new_mols))]

        for i in range(len(new_mols)):
            (molSmiles, neutralised) = NeutraliseCharges_RemoveSalt(all_mols[i])
            if molSmiles is not None:
                new_mols[i] = molSmiles
            else:
                valid_filter[i] = False
            if i % 100 == 0:
                print(i)

        custom_compound_df['canonical_smiles'] = new_mols
        N_raw = len(custom_compound_df)
        custom_compound_df = custom_compound_df.loc[valid_filter]
        N_clean = len(custom_compound_df)
        #custom_compound_df['chembl_id'] = custom_compound_df[custom_id_field].astype(int)
        custom_compound_df['chembl_id'] = custom_compound_df[custom_id_field]
        print('Read in %d compounds; %d valid compounds.' % (N_raw, N_clean))

        if FP_type == 'Circular_2048':
            custom_compound_df.to_csv(dataset_file)
            featurizer = dc.feat.CircularFingerprint(size=2048)
            loader = dc.data.CSVLoader(id_field=custom_id_field,
                                   smiles_field='canonical_smiles',
                                   tasks = [custom_task_field],
                                   featurizer=featurizer)
            custom_dataset = loader.featurize(dataset_file)
        N_custom = custom_compound_df.shape[0]

    # load prev_model
    print('------------- Loading previous trained models ------------------')
    model = dc.models.RobustMultitaskRegressor(n_tasks = n_tasks, n_features = n_features, layer_sizes = layer_sizes,
                                               bypass_layer_sizes= bypass_layer_sizes, bypass_dropouts = bypass_dropouts,
                                               dropout = dropout, learning_rate = learning_rate)
    model.restore(checkpoint = prev_model)

    ### for chemical information
    print('------------- Prepare information for chemicals ------------------')
    # build transfer model
    model_transfer = calculate_transfer_values(prev_model=prev_model, n_tasks = n_tasks,
                                           layer_sizes = layer_sizes, bypass_layer_sizes=bypass_layer_sizes, n_layer = n_layer)
    if custom_file_flag:
        values = np.concatenate((dataset.X, custom_dataset.X), axis = 0)
    else:
        values = dataset.X
    coord_values = dim_reduce(model_transfer, values)

    # calculate predictions
    pred_mat = model.predict(dataset)
    pred_mat = pred_mat[:,valid_mask]

    pred_df = pd.DataFrame(pred_mat)
    pred_df.columns = ['pred_' + xx for xx in model_tasks[valid_mask]]
    pred_df['chembl_id'] = dataset.ids

    coord_df = pd.DataFrame(coord_values[0:N_training,:])
    coord_df.columns = ['x', 'y']
    coord_df['chembl_id'] = dataset.ids

    coord_df['chembl_id'] = coord_df['chembl_id'].astype(int)
    pred_df['chembl_id'] = pred_df['chembl_id'].astype(int)
    compound_df = pd.merge(df, coord_df, on = 'chembl_id')
    compound_df = pd.merge(compound_df, pred_df, on = 'chembl_id')

    if custom_file_flag:
        pred_custom_mat = model.predict(custom_dataset)
        pred_custom_mat = pred_custom_mat[:,valid_mask]

        pred_df2 = pd.DataFrame(pred_custom_mat)
        pred_df2.columns = ['pred_' + xx for xx in model_tasks[valid_mask]]
        pred_df2['chembl_id'] = custom_dataset.ids

        coord_df2 = pd.DataFrame(coord_values[N_training:coord_values.shape[0],:])
        coord_df2.columns = ['x', 'y']
        coord_df2['chembl_id'] = custom_dataset.ids

        if not type(custom_compound_df[custom_id_field].iloc[0]) == str:
            coord_df2['chembl_id'] = coord_df2['chembl_id'].astype(int)
            pred_df2['chembl_id'] = pred_df2['chembl_id'].astype(int)
        compound_df2 = pd.merge(custom_compound_df, coord_df2, on = 'chembl_id')
        compound_df2 = pd.merge(compound_df2, pred_df2, on = 'chembl_id')

    ### for cluster information (including the custom datasets?)
    print('------------- Prepare information for minibatches ------------------')
    if custom_file_flag:
        pred_mat_concat = np.concatenate((pred_mat, pred_custom_mat), axis = 0)
    else:
        pred_mat_concat = pred_mat
    mbk = cluster_MiniBatch(coord_values) # or values?

    mbk.means_labels_unique = np.unique(mbk.labels_)
    n_row = len(np.unique(mbk.labels_))
    n_col = pred_mat.shape[1]
    cluster_info_mat = np.zeros((n_row, (n_col + 3)))
    for k in range(n_row):
        mask = mbk.labels_ == mbk.means_labels_unique[k]
        cluster_info_mat[k, 0:n_col] = np.nanmean(pred_mat_concat[mask,:], axis = 0)
        cluster_info_mat[k, n_col] = sum(mask)
        cluster_info_mat[k, (n_col + 1) : (n_col + 3)] = np.nanmean(coord_values[mask,:], axis = 0)
    compound_df['label'] = mbk.labels_[0:N_training]
    batch_df = pd.DataFrame(cluster_info_mat)
    batch_df.columns = ['avg_' + xx for xx in model_tasks[valid_mask]] + ['size', 'coordx', 'coordy']
    batch_df['Label_id'] = mbk.means_labels_unique

    if custom_file_flag:
        compound_df2['label'] = mbk.labels_[N_training: len(mbk.labels_)]

    ### for task information
    print('------------- Prepare information for tasks ------------------')
    TASK_LAYERS = ['Dense_%d/Dense_%d/Relu:0' % (10 + n_bypass * 2 * idx, 10 + n_bypass * 2 * idx)
               for idx in range(n_tasks)]
    TASK_LAYERS = list(np.array(TASK_LAYERS)[valid_mask])
    SHARE_LAYER = 'Dense_7/Dense_7/Relu:0'
    grad_mat = np.zeros((len(TASK_LAYERS)+1, n_features))

    r2_list = []
    for i in range(len(TASK_LAYERS)):
        grad_mat[i,:] = calculate_gradients_RobustMT(dataset.X, TASK_LAYERS[i], prev_model,
                                                 n_tasks, n_features, layer_sizes, bypass_layer_sizes)
        #r2_list.append(pearsonr(dataset.y[:,i], pred_mat[:,i])[0] ** 2)
    grad_mat[len(TASK_LAYERS),:] = calculate_gradients_RobustMT(dataset.X, SHARE_LAYER, prev_model,
                                                           n_tasks, n_features, layer_sizes, bypass_layer_sizes)
    task_df = pd.DataFrame(grad_mat.T)
    task_df.columns = list(model_tasks[valid_mask]) + ['SHARE']

    ### generate color labels by default
    print('------- Generate color labels with default K of %d --------' % np.min([n_tasks, K]))
    batch_df, task_df, compound_df = update_bicluster(batch_df, task_df, compound_df, K = K)

    if custom_file_flag:
        lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
        lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
        lut222 = dict(zip(compound_df['label'], compound_df['label_color']))
        compound_df2['batch_label_color'] = compound_df2['label'].map(lut2)
        compound_df2['batch_label'] = compound_df2['label'].map(lut22)
        compound_df2['label_color'] = compound_df2['label'].map(lut222)

    ### wrapping up
    print('-------------- Saving datasets ----------------')
    compound_df.to_csv(output_prefix + 'compound_df.csv', index = False)
    batch_df.to_csv(output_prefix + 'batch_df.csv', index = False)
    task_df.to_csv(output_prefix + 'task_df.csv', index = False)

    if custom_file_flag:
        compound_df2.to_csv(output_prefix + 'compound_custom_df.csv', index = False)

    return


def generate_RUNKEY_dataframe_baseline(output_prefix, task, dataset_file, FP_type,
                                       add_features, mode = 'SVR',
                                       MT_dat_name = './data/MT_data_clean_June28.csv',
                                       smiles_field = 'canonical_smiles', id_field = 'chembl_id',
                                       custom_file = None, custom_id_field = None, custom_task_field = None,
                                       custom_smiles_field = None, sep_custom_file = ','):

    print('------------- Loading dataset and train baseline model --------------------')
    dataset, df = prepare_dataset(MT_dat_name, [task], dataset_file, FP_type,
                                  smiles_field = smiles_field,
                                  add_features = add_features,
                                  id_field = id_field, model_flag = 'ST')
    df['chembl_id'] = df[id_field].astype(int)
    switch_field = lambda item:'canonical_smiles' if  item == smiles_field  else item
    df.columns = [switch_field(item) for item in df.columns.tolist()]
    N_training = df.shape[0]

    # if custom file exisit, processing the file
    custom_file_flag = False
    if custom_file is not None:
        custom_file_flag = True
        print('------------- Loading custom file --------------------')
        custom_compound_df = pd.read_csv(custom_file, sep = sep_custom_file)

        ## checking point: the ids for custom files must be unique
        if len(set(custom_compound_df[custom_id_field])) < custom_compound_df.shape[0]:
            print('Error: The ids for custom files must be unique')
            return

        all_mols = custom_compound_df[custom_smiles_field].tolist()
        new_mols = all_mols
        valid_filter = [True for k in range(len(new_mols))]

        for i in range(len(new_mols)):
            (molSmiles, neutralised) = NeutraliseCharges_RemoveSalt(all_mols[i])
            if molSmiles is not None:
                new_mols[i] = molSmiles
            else:
                valid_filter[i] = False
            if i % 100 == 0:
                print(i)

        custom_compound_df['canonical_smiles'] = new_mols
        N_raw = len(custom_compound_df)
        custom_compound_df = custom_compound_df.loc[valid_filter]
        N_clean = len(custom_compound_df)
        #custom_compound_df['chembl_id'] = custom_compound_df[custom_id_field].astype(int)
        custom_compound_df['chembl_id'] = custom_compound_df[custom_id_field]
        print('Read in %d compounds; %d valid compounds.' % (N_raw, N_clean))

        if FP_type == 'Circular_2048':
            custom_compound_df.to_csv(dataset_file)
            featurizer = dc.feat.CircularFingerprint(size=2048)
            loader = dc.data.CSVLoader(id_field=custom_id_field,
                                   smiles_field='canonical_smiles',
                                   tasks = [custom_task_field],
                                   featurizer=featurizer)
            custom_dataset = loader.featurize(dataset_file)
        N_custom = custom_compound_df.shape[0]

    # split training/test dataset
    splitter = dc.splits.RandomSplitter(dataset_file)
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
    # baseline model training
    X_new = np.c_[[1]*train_dataset.X.shape[0], train_dataset.X]
    if mode == 'SVR':
        clf = LinearSVR(C = 1.0, epsilon = 0.2)
        try:
            clf.fit(X_new, train_dataset.y)
        except Exception as e:
            print(e)
    elif mode == 'RidgeCV':
        alphas = np.logspace(start = -1, stop = 2, num = 20)
        clf = linear_model.RidgeCV(alphas)
        try:
            clf.fit(X_new, train_dataset.y)
        except Exception as e:
            print(e)

    print('------------- Prepare information for chemicals ------------------')
    if custom_file_flag:
        values = np.concatenate((dataset.X, custom_dataset.X), axis = 0)
    else:
        values = dataset.X

    pca = PCA(n_components = 20)
    value_reduced_20d = pca.fit_transform(values)
    tsne = TSNE(n_components = 2)
    coord_values = tsne.fit_transform(value_reduced_20d)

    pred_mat = clf.predict(np.c_[[1]*dataset.X.shape[0], dataset.X])

    pred_df = pd.DataFrame(pred_mat)
    pred_df.columns = ['pred_' + task]
    pred_df['chembl_id'] = dataset.ids

    coord_df = pd.DataFrame(coord_values[0:N_training,:])
    coord_df.columns = ['x','y']
    coord_df['chembl_id'] = dataset.ids

    pred_df['chembl_id'] = pred_df['chembl_id'].astype(int)
    coord_df['chembl_id'] = coord_df['chembl_id'].astype(int)
    compound_df = pd.merge(df, coord_df, on = 'chembl_id')
    compound_df = pd.merge(compound_df, pred_df, on = 'chembl_id')

    if custom_file_flag:
        pred_custom_mat = clf.predict(np.c_[[1]*custom_dataset.X.shape[0], custom_dataset.X])

        pred_df2 = pd.DataFrame(pred_custom_mat)
        pred_df2.columns = ['pred_' + task]
        pred_df2['chembl_id'] = custom_dataset.ids

        coord_df2 = pd.DataFrame(coord_values[N_training:coord_values.shape[0],:])
        coord_df2.columns = ['x', 'y']
        coord_df2['chembl_id'] = custom_dataset.ids

        if not type(custom_compound_df[custom_id_field].iloc[0]) == str:
            coord_df2['chembl_id'] = coord_df2['chembl_id'].astype(int)
            pred_df2['chembl_id'] = pred_df2['chembl_id'].astype(int)
        compound_df2 = pd.merge(custom_compound_df, coord_df2, on = 'chembl_id')
        compound_df2 = pd.merge(compound_df2, pred_df2, on = 'chembl_id')

    print('------------- Prepare information for minibatches ------------------')
    if custom_file_flag:
        pred_mat_concat = np.concatenate((pred_mat, pred_custom_mat), axis = 0)
    else:
        pred_mat_concat = pred_mat

    mbk = cluster_MiniBatch(coord_values)
    mbk.means_labels_unique = np.unique(mbk.labels_)
    n_row = len(mbk.means_labels_unique)
    n_col = 1
    cluster_info_mat = np.zeros((n_row, (n_col + 3)))

    for k in range(n_row):
        mask = mbk.labels_ == mbk.means_labels_unique[k]
        cluster_info_mat[k, 0:n_col] = np.nanmean(pred_mat_concat[mask], axis = 0)
        cluster_info_mat[k, n_col] = sum(mask)
        cluster_info_mat[k, (n_col + 1) : (n_col + 3)] = np.nanmean(coord_values[mask,:], axis = 0)
    compound_df['label'] = mbk.labels_[0:N_training]
    batch_df = pd.DataFrame(cluster_info_mat)
    batch_df.columns = ['avg_' + task] + ['size', 'coordx', 'coordy']
    batch_df['Label_id'] = mbk.means_labels_unique

    if custom_file_flag:
        compound_df2['label'] = mbk.labels_[N_training: len(mbk.labels_)]

    print('------------- Saving datasets --------------')
    grad_mat = clf.coef_.reshape(-1)[1:]
    task_df = pd.DataFrame(grad_mat.T)
    task_df.columns = [task]

    batch_df, task_df, compound_df = update_bicluster(batch_df, task_df, compound_df, mode = 'ST')

    if custom_file_flag:
        lut2 = dict(zip(batch_df['Label_id'], batch_df['batch_label_color']))
        lut22 = dict(zip(batch_df['Label_id'], batch_df['batch_label']))
        lut222 = dict(zip(compound_df['label'], compound_df['label_color']))
        compound_df2['batch_label_color'] = compound_df2['label'].map(lut2)
        compound_df2['batch_label'] = compound_df2['label'].map(lut22)
        compound_df2['label_color'] = compound_df2['label'].map(lut222)

    compound_df.to_csv(output_prefix + 'compound_df.csv', index = False)
    batch_df.to_csv(output_prefix + 'batch_df.csv', index = False)
    task_df.to_csv(output_prefix + 'task_df.csv', index = False)

    if custom_file_flag:
        compound_df2.to_csv(output_prefix + 'compound_custom_df.csv', index = False)
    return
