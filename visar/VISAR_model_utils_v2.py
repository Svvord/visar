import deepchem as dc
import numpy as np
import pandas as pd
from keras.models import load_model

#from model_training_utils_v2 import prepare_dataset
#from model_landscape_utils_v2 import calculate_transfer_values, calculate_transfer_values_ST, dim_reduce, cluster_MiniBatch
#from model_SAR_utils_v2 import calculate_gradients_RobustMT, calculate_gradients_ST

from scipy.stats import pearsonr
from bokeh.palettes import Category20_20, Category20b_20

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn import preprocessing

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

def generate_RUNKEY_dataframe_RobustMT(prev_model, output_prefix, task_list, dataset_file, FP_type, add_features,
                              n_features, layer_sizes, bypass_layer_sizes, model_flag, n_bypass,
                              MT_dat_name = './data/MT_data_clean_June28.csv',
                              smiles_field = 'canonical_smiles', id_field = 'chembl_id',
                              bypass_dropouts = [.5], dropout = 0.5, learning_rate = 0.001, n_layer = 2):
    if add_features is None:
    	tasks = task_list
    else:
    	tasks = task_list + add_features

    n_tasks = len(tasks)
    # load dataset
    print('------------- Loading dataset --------------------')
    dataset, df = model_training_utils_v2.prepare_dataset(MT_dat_name, task_list, dataset_file, FP_type,
                                  smiles_field = smiles_field,
                                  add_features = add_features,
                                  id_field = id_field, model_flag = model_flag)

    # load prev_model
    print('------------- Loading previous trained models ------------------')
    model = dc.models.RobustMultitaskRegressor(n_tasks = n_tasks, n_features = n_features, layer_sizes = layer_sizes,
                                               bypass_layer_sizes= bypass_layer_sizes, bypass_dropouts = bypass_dropouts,
                                               dropout = dropout, learning_rate = learning_rate)
    model.restore(checkpoint = prev_model)

    ### for chemical information
    print('------------- Prepare information for chemicals ------------------')
    # build transfer model
    model_transfer = model_landscape_utils_v2.calculate_transfer_values(prev_model=prev_model, n_tasks = n_tasks,
                                           layer_sizes = layer_sizes, bypass_layer_sizes=bypass_layer_sizes, n_layer = n_layer)
    coord_values = model_landscape_utils_v2.dim_reduce(model_transfer,dataset.X)
    pred_mat = model.predict(dataset)

    pred_df = pd.DataFrame(pred_mat)
    pred_df.columns = ['pred_' + xx for xx in tasks]
    pred_df['chembl_id'] = dataset.ids

    coord_df = pd.DataFrame(coord_values)
    coord_df.columns = ['x', 'y']
    coord_df['chembl_id'] = dataset.ids

    compound_df = pd.merge(df, coord_df, on = 'chembl_id')
    compound_df = pd.merge(compound_df, pred_df, on = 'chembl_id')

    ### for cluster information
    print('------------- Prepare information for minibatches ------------------')
    transfer_values = model_transfer.predict(dataset.X)
    mbk = model_landscape_utils_v2.cluster_MiniBatch(transfer_values)
    mbk.means_labels_unique = np.unique(mbk.labels_)
    n_row = len(np.unique(mbk.labels_))
    n_col = pred_mat.shape[1]
    cluster_info_mat = np.zeros((n_row, (n_col + 3)))
    for k in range(n_row):
        mask = mbk.labels_ == mbk.means_labels_unique[k]
        cluster_info_mat[k, 0:n_col] = np.nanmean(pred_mat[mask,:], axis = 0)
        cluster_info_mat[k, n_col] = sum(mask)
        cluster_info_mat[k, (n_col + 1) : (n_col + 3)] = np.nanmean(coord_values[mask,:], axis = 0)
    compound_df['label'] = mbk.labels_
    batch_df = pd.DataFrame(cluster_info_mat)
    batch_df.columns = ['avg_' + xx for xx in tasks] + ['size', 'coordx', 'coordy']
    batch_df['Label_id'] = mbk.means_labels_unique

    ### for task information
    print('------------- Prepare information for tasks ------------------')
    TASK_LAYERS = ['Dense_%d/Dense_%d/Relu:0' % (10 + n_bypass * 2 * idx, 10 + n_bypass * 2 * idx)
               for idx in range(n_tasks)]
    SHARE_LAYER = 'Dense_7/Dense_7/Relu:0'
    grad_mat = np.zeros((len(TASK_LAYERS)+1, n_features))

    r2_list = []
    for i in range(len(TASK_LAYERS)):
        grad_mat[i,:] = model_SAR_utils_v2.calculate_gradients_RobustMT(dataset.X, TASK_LAYERS[i], prev_model,
                                                 n_tasks, n_features, layer_sizes, bypass_layer_sizes)
        #r2_list.append(pearsonr(dataset.y[:,i], pred_mat[:,i])[0] ** 2)
    grad_mat[len(TASK_LAYERS),:] = model_SAR_utils_v2.calculate_gradients_RobustMT(dataset.X, SHARE_LAYER, prev_model,
                                                           n_tasks, n_features, layer_sizes, bypass_layer_sizes)
    task_df = pd.DataFrame(grad_mat.T)
    task_df.columns = tasks + ['SHARE']

    ### generate color labels by default
    print('------- Generate color labels with default K of %d --------' % np.min([n_tasks, 20]))
    batch_df, task_df, compound_df = update_bicluster(batch_df, task_df, compound_df)

    ### wrapping up
    print('-------------- Saving datasets ----------------')
    compound_df.to_csv(output_prefix + 'compound_df.csv', index = False)
    batch_df.to_csv(output_prefix + 'batch_df.csv', index = False)
    task_df.to_csv(output_prefix + 'task_df.csv', index = False)

    return


def generate_RUNKEY_dataframe_ST(prev_model, output_prefix, task_list, dataset_file, FP_type,
                                 add_features, mode = 'ST',
                                 MT_dat_name = './data/MT_data_clean_June28.csv', n_layer = 2,
                                 smiles_field = 'canonical_smiles', id_field = 'chembl_id'):
    if add_features is None:
        tasks = task_list
    else:
        tasks = task_list + add_features

    # load dataset
    print('------------- Loading dataset --------------------')
    dataset, df = model_training_utils_v2.prepare_dataset(MT_dat_name, task_list, dataset_file, FP_type,
                                  smiles_field = smiles_field,
                                  add_features = add_features,
                                  id_field = id_field, model_flag = 'ST')
    # load prev_model
    print('------------- Loading previous trained models ------------------')
    model = load_model(prev_model)

    ### for chemical information
    print('------------- Prepare information for chemicals ------------------')
    # build transfer model
    model_transfer = model_landscape_utils_v2.calculate_transfer_values_ST(prev_model=prev_model, n_layer = n_layer)
    coord_values = model_landscape_utils_v2.dim_reduce(model_transfer,dataset.X)
    pred_mat = model.predict(dataset.X)

    pred_df = pd.DataFrame(pred_mat)
    pred_df.columns = ['pred_' + xx for xx in tasks]
    pred_df['chembl_id'] = dataset.ids

    coord_df = pd.DataFrame(coord_values)
    coord_df.columns = ['x', 'y']
    coord_df['chembl_id'] = dataset.ids

    compound_df = pd.merge(df, coord_df, on = 'chembl_id')
    compound_df = pd.merge(compound_df, pred_df, on = 'chembl_id')

    ### for cluster information
    print('------------- Prepare information for minibatches ------------------')
    transfer_values = model_transfer.predict(dataset.X)
    mbk = model_landscape_utils_v2.cluster_MiniBatch(transfer_values)
    mbk.means_labels_unique = np.unique(mbk.labels_)
    n_row = len(np.unique(mbk.labels_))
    n_col = pred_mat.shape[1]
    cluster_info_mat = np.zeros((n_row, (n_col + 3)))
    for k in range(n_row):
        mask = mbk.labels_ == mbk.means_labels_unique[k]
        cluster_info_mat[k, 0:n_col] = np.nanmean(pred_mat[mask,:], axis = 0)
        cluster_info_mat[k, n_col] = sum(mask)
        cluster_info_mat[k, (n_col + 1) : (n_col + 3)] = np.nanmean(coord_values[mask,:], axis = 0)
    compound_df['label'] = mbk.labels_
    batch_df = pd.DataFrame(cluster_info_mat)
    batch_df.columns = ['avg_' + xx for xx in tasks] + ['size', 'coordx', 'coordy']
    batch_df['Label_id'] = mbk.means_labels_unique

    ### for task information
    print('------------- Prepare information for tasks ------------------')

    grad_mat = model_SAR_utils_v2.calculate_gradients_ST(dataset.X, prev_model)
    task_df = pd.DataFrame(grad_mat)

    ### generate color labels by default
    print('------- Generate color labels with default K of 5 --------')
    batch_df, task_df, compound_df = update_bicluster(batch_df, task_df, compound_df, mode = mode)

    ### wrapping up
    print('-------------- Saving datasets ----------------')
    compound_df.to_csv(output_prefix + 'compound_df.csv', index = False)
    batch_df.to_csv(output_prefix + 'batch_df.csv', index = False)
    task_df.to_csv(output_prefix + 'task_df.csv', index = False)

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

def generate_performance_plot_RobustMT(train_file, test_file, task_list):
    # load data
    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(train_file)

    # perpare dataframe for plotting
    test_df.columns = ['step'] + task_list
    train_df.columns = ['step'] + task_list

    test_df = test_df.set_index('step')
    test_df.columns.name = 'tasks'
    plot_test = pd.DataFrame(test_df.stack(), columns = ['R2']).reset_index()
    plot_test['tt'] = 'test'

    train_df = train_df.set_index('step')
    train_df.columns.name = 'tasks'
    plot_train = pd.DataFrame(train_df.stack(), columns = ['R2']).reset_index()
    plot_train['tt'] = 'train'

    plot_df = pd.concat([plot_train, plot_test])

    return plot_df
