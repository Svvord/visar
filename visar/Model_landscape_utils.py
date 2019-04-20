# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:13:30 2019

@author: Dingqy
"""

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from itertools import cycle
import matplotlib.colors as colors

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import PandasTools
import cairosvg

from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource

from Model_SAR_viz_utils import calculate_gradients_ST, moltosvg, color_rendering, gradient2atom
from Model_training_utils import prepare_dataset, extract_clean_dataset

#------------------------------------------------
# functions for transfer value calculating (layer1/2/3/4)
def ST_model_layer1(n_features, layer_size, pretrained_params,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5):
    training_dat = Input(shape = (n_features,), dtype = 'float32')
    X = Dense(layer_size[0], activation = 'relu')(training_dat)
    model = Model(input = training_dat, output = X)
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    model.compile(loss = 'mean_squared_error', optimizer=optimizer)
    
    # set parameters
    model.layers[1].set_weights(pretrained_params[0])
    return model

def ST_model_layer2(n_features, layer_size,pretrained_params,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5):
    training_dat = Input(shape = (n_features,), dtype = 'float32')
    X = Dense(layer_size[0], activation = 'relu')(training_dat)
    X = Dense(layer_size[1], activation = 'relu')(X)
    model = Model(input = training_dat, output = X)
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    model.compile(loss = 'mean_squared_error', optimizer=optimizer)
    
    # set parameters
    model.layers[1].set_weights(pretrained_params[0])
    model.layers[2].set_weights(pretrained_params[1])
    return model

def ST_model_layer3(n_features, layer_size, pretrained_params,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5):
    training_dat = Input(shape = (n_features,), dtype = 'float32')
    X = Dense(layer_size[0], activation = 'relu')(training_dat)
    X = Dense(layer_size[1], activation = 'relu')(X)
    X = Dense(layer_size[2], activation = 'relu')(X)
    model = Model(input = training_dat, output = X)
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    model.compile(loss = 'mean_squared_error', optimizer=optimizer)
    
    # set parameters
    model.layers[1].set_weights(pretrained_params[0])
    model.layers[2].set_weights(pretrained_params[1])
    model.layers[3].set_weights(pretrained_params[2])
    return model

def ST_model_layer4(n_features, layer_size, pretrained_params,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5):
    training_dat = Input(shape = (n_features,), dtype = 'float32')
    X = Dense(layer_size[0], activation = 'relu')(training_dat)
    X = Dense(layer_size[1], activation = 'relu')(X)
    X = Dense(layer_size[2], activation = 'relu')(X)
    X = Dense(layer_size[3], activation = 'relu')(X)
    model = Model(input = training_dat, output = X)
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    model.compile(loss = 'mean_squared_error', optimizer=optimizer)
    
    # set parameters
    model.layers[1].set_weights(pretrained_params[0])
    model.layers[2].set_weights(pretrained_params[1])
    model.layers[3].set_weights(pretrained_params[2])
    model.layers[4].set_weights(pretrained_params[3])
    return model

def calculate_transfer_values(prev_model, n_layer):
    model = load_model(prev_model)
    layer1 = model.layers[1]
    param1 = layer1.get_weights()
    layer2 = model.layers[3]
    param2 = layer2.get_weights()
    
    n_features = param1[0].shape[0]
    layer_size = [param1[0].shape[1], param2[0].shape[1]]
    
    if n_layer == 1:
        transfer_model = ST_model_layer1(n_features, layer_size, [param1, param2])
    elif n_layer == 2:
        transfer_model = ST_model_layer2(n_features, layer_size, [param1, param2])
    elif n_layer == 3:
        layer3 = model.layers[5]
        param3 = layer3.get_weights()
        layer_size = [param1[0].shape[1], param2[0].shape[1], param3[0].shape[1]]
        transfer_model = ST_model_layer3(n_features, layer_size, [param1, param2, param3])
    elif n_layer == 4:
        layer3 = model.layers[5]
        param3 = layer3.get_weights()
        layer4 = model.layers[7]
        param4 = layer4.get_weights()
        layer_size = [param1[0].shape[1], param2[0].shape[1], 
                      param3[0].shape[1], param4[0].shape[1]]
        transfer_model = ST_model_layer4(n_features, layer_size, 
                                         [param1, param2, param3, param4])
    return transfer_model

#-----------------------------------------------
def dim_reduce(transfer_model, X):
    transfer_values = transfer_model.predict(X)
    pca = PCA(n_components = 20)
    value_reduced_20d = pca.fit_transform(transfer_values)
    tsne = TSNE(n_components = 2)
    value_reduced = tsne.fit_transform(value_reduced_20d)
    return value_reduced

#------------------------------------------------
def plot_scatter(values, cls, output_name):
    '''
    input: values --- coordinates of each chemical (2 columns)
           cls --- the value for each chemical
           output_name --- name of the output figure

    create a color-map with a different color for each class
    '''
    import matplotlib.cm as cm
    cmap = cm.RdBu_r
    # Get the color for each sample
    normalize = matplotlib.colors.Normalize(vmin=min(cls), vmax=max(cls))
    colors = [cmap(normalize(value)) for value in cls]
    # Extract the x- and y-values
    x = values[:, 0]
    y = values[:, 1]
    # plot
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(x, y, color=colors)
    cax, _ = matplotlib.colorbar.make_axes(ax)
    savefig(output_name)
    plt.show()

#-------------------------------------------------
def cluster_MiniBatch(values, grain_size = 30):
    # clustering using KMeans minibatch algorithm
    n_clusters = int(values.shape[0] / grain_size)
    mbk = MiniBatchKMeans(init = 'k-means++', n_clusters = n_clusters, batch_size=100,
                      n_init = 10, max_no_improvement=10, verbose=0, random_state=0)
    mbk.fit(values)
    return mbk

def plot_clusters(mbk, values):
    mbk.means_labels_unique = np.unique(mbk.labels_)
    colors_ = cycle(colors.cnames.keys())
    n_clusters = mbk.means_labels_unique.size
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1, 1, 1)
    for this_centroid, k, col in zip(mbk.cluster_centers_,
                                     range(n_clusters), colors_):
        mask = mbk.labels_ == k
        ax.scatter(values[mask, 0], values[mask, 1], marker='.',
                   c='w', edgecolor=col, alpha=0.5)
        ax.scatter(this_centroid[0], this_centroid[1], marker='+',
                   c='k', s=25)
    ax.set_title("MiniBatchKMeans")
    return

#--------------------------------------------------
def pack_clusters(smiles, labels, molregno, acts, output_prefix, values, cutoff = 4.5):
    '''
    generate sdf files of active clusters for further analysis
    input: smiles --- list of SMILES
           labels --- list of labels (derived from minibatch clustering)
           molregno --- list of chemical ids
           acts --- list of values for chemicals
           output_prefix --- string defining the saving directory and file name prefix
           values --- coordinates of the chemicals (2 columns)
    output: saving sdf files in the directory of destination
    '''
    df = pd.DataFrame({'smi': smiles, 'label': labels, 'molregno': molregno,
                 'Act': acts, 'coord1': values[:,0], 'coord2': values[:,1]})
    df.to_csv(output_prefix + 'store.csv')
    PandasTools.AddMoleculeColumnToFrame(df,'smi','ROMol')
    
    unique_labels = list(set(labels))
    for i in range(len(unique_labels)):
        mask = [item == unique_labels[i] for item in labels]
        y_mean = np.mean(df['Act'].loc[mask])
        if y_mean >= cutoff:
            sub_df = df.loc[mask]
            sdf_file = output_prefix + 'Label_%d.sdf' % unique_labels[i]
            PandasTools.WriteSDF(sub_df, sdf_file, idName='molregno', properties=sub_df.columns)
            print('%d compounds with label %d of mean %.4f: ' % (len(sub_df), unique_labels[i], y_mean))

#-------------------------------------------------
def generate_sdf(df, id_var, fname_right, on_var_right, smi_var_right, 
                 fname_left, on_var_left, smi_var_left, output_sdf_name):
    '''
    merge dataframe containing chemical information and custom dataframe with chemicals of interest, 
              and pack them together in sdf file for further analysis
    input: df --- dataframe containing calculated coordinates for both training dataset and custom dataset
           id_var -- idenifier for df
           fname_right -- name of raw dataset
           on_var_right -- merge identifier for raw dataset
           smi_var_right -- SMILES field for raw dataset
           fname_left -- name of custom dataset
           on_var_left -- merge identifier for custom dataset
           smi_var_left -- SMILES field for custom dataset
           output_sdf_name
    output: sdf_file
    '''

    # prepare right df
    raw_df = pd.read_csv(fname_right)
    info_df = pd.DataFrame({'id': raw_df[on_var_right],
                            'smi': raw_df[smi_var_right]})
    # prepare left df
    left_info_df = pd.read_csv(fname_left)
    left_info_df = pd.DataFrame({'id': left_info_df[on_var_left],
                                 'smi': left_info_df[smi_var_left]})
    info_df = pd.concat([info_df, left_info_df], axis = 0)
    # merge
    new_df = pd.merge(df, info_df, left_on = id_var, right_on = 'id')
    PandasTools.AddMoleculeColumnToFrame(new_df,'smi','ROMol')
    PandasTools.WriteSDF(new_df, output_sdf_name, idName='id', properties=df.columns)
    return new_df

#------------------------------------------------
def sdf2df(sdf_file):
    '''
    convert sdf file to the type of pd.DataFrame
    '''
    suppl = Chem.SDMolSupplier(sdf_file)
    mols = [mol for mol in suppl]
    mol_names = [mol.GetProp('_Name') for mol in mols]
    df = pd.DataFrame({'molregno': mol_names})

    props = list(mols[0].GetPropsAsDict().keys())
    for prop in props:
        values = [mol.GetProp(prop) for mol in mols]
        df[prop] = values
    return df

def df2sdf(df, output_sdf_name, smiles_field, id_field, custom_filter = None):
    '''
    pack pd.DataFrame to sdf_file
    '''
    if custom_filter:
        df = df.loc[custom_filter]
    PandasTools.AddMoleculeColumnToFrame(df,smiles_field,'ROMol')
    PandasTools.WriteSDF(df, output_sdf_name, idName=id_field, properties=df.columns)

    return

def SAR_rendering(X, prev_model, df, id_field, smiles_field, SAR_result_dir, vis_cutoff = 50):
    gradients = calculate_gradients_ST(X, prev_model)
    for k in range(len(df)):
        gradient0 = gradients[k, :]
        smi = df[smiles_field].iloc[k]
        mol, highlit_pos, highlit_neg, atomsToUse = gradient2atom(smi, gradient0)
        atomsToUse, color_dict = color_rendering(atomsToUse, vis_cutoff)
        img = moltosvg(mol, molSize=(450,300), highlightAtoms=[m for m in range(len(atomsToUse))],
            highlightAtomColors = color_dict,highlightBonds=[])
        cairosvg.svg2png(bytestring=img.data, write_to = SAR_result_dir + '%s_img.png' % str(df[id_field].iloc[k]))
    png_file_names = [SAR_result_dir + str(item) + '_img.png' for item in df[id_field].tolist()]
    df['imgs'] = png_file_names
    return df

def interactive_plot(plot_df, x_column, y_column, color_column, id_field, value_field, label_field):
    # color rendering
    from bokeh.palettes import RdBu11
    COLORS = RdBu11
    N_COLORS = len(COLORS)
    groups = pd.qcut(plot_df[color_column].tolist(), N_COLORS, duplicates='drop')
    c = [COLORS[xx] for xx in groups.codes]
    plot_df['c'] = c

    # prepare bokeh dataset
    cds_df = ColumnDataSource(plot_df)

    # set hover display format
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@imgs" height="300" alt="@imgs" width="240"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@%s</span>
                <span style="font-size: 15px; color: #966;">[$index]</span>
            </div>
            <div>
                <span style="font-size: 10px;">Activity:</span>
                <span>@%s</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cluster label:</span>
                <span>@%s</span>
            </div>
            <div>
                <span style="font-size: 15px;">Location</span>
                <span style="font-size: 10px; color: #696;">($x, $y)</span>
            </div>
        </div>
    """ % (id_field, value_field, label_field)
    p = figure(plot_width = 800, plot_height = 800, toolbar_location = 'below',
               tools = 'pan,box_zoom,reset,lasso_select,save,hover', tooltips=TOOLTIPS)
    p.circle(source = cds_df, x = x_column, y = y_column, color='c', size=9)
    return p

#================================================
def landscape_building(task_name, db_name, log_path, FP_type,
                       prev_model, n_layer, 
                       SAR_result_dir, output_sdf_name,
                       pack_sdf = True, vis_cutoff = 50,
                       smiles_field = 'salt_removed_smi', id_field = 'molregno'):
    '''
    generate chemical landscape for a specific task
    '''
    # step1: prepare dataset for the landscape
    print('==== preparing dataset ... ====')
    dataset_file = '%s/temp.csv' % (log_path)
    dataset, df = prepare_dataset(db_name, [task_name], dataset_file, FP_type,
                                  smiles_field = smiles_field, id_field = id_field)

    # step2: calculate transfer value
    print('==== calculating transfer values ... ====')
    transfer_model = calculate_transfer_values(prev_model, n_layer)
    value_reduced = dim_reduce(transfer_model, dataset.X)
    df['coord1'] = value_reduced[:,0]
    df['coord2'] = value_reduced[:,1]

    # step3: MiniBatch clustering
    mbk = cluster_MiniBatch(value_reduced)
    df['Label'] = mbk.labels_

    # step4: SAR rendering for all compounds
    print('==== rendering SAR for chemicals on the landscape ... ====')
    df = SAR_rendering(dataset.X, prev_model, df, id_field, smiles_field, SAR_result_dir, vis_cutoff)

    # step5: pack sdf
    if pack_sdf:
        print('==== packing sdf file ... ====')
        df2sdf(df, output_sdf_name, smiles_field, id_field)

    # step6: build interactive plot using Bokeh
    plot_df = df.drop(['ROMol'], axis = 1)
    return plot_df


def landscape_positioning(custom_file, custom_smi_field, custom_id_field, custom_task_field,
                          landscape_sdf, task_name, db_name, FP_type, log_path,
                          prev_model, n_layer, custom_SAR_result_dir, custom_sdf_name,
                          pack_sdf = True, vis_cutoff = 50,
                          smiles_field = 'salt_removed_smi', id_field = 'molregno'):
    # step1: prepare dataset for the landscape
    print('==== preparing dataset ... ====')
    df = sdf2df(landscape_sdf)
    df_imgs = pd.DataFrame({'molregno': [int(item) for item in df['molregno'].tolist()], 'imgs': df['imgs'].tolist()})

    dataset_file = '%s/temp.csv' % (log_path)
    dataset, df = prepare_dataset(db_name, [task_name], dataset_file, FP_type,
                                  smiles_field = smiles_field, id_field = id_field)
    df = pd.merge(df, df_imgs, right_on = 'molregno', left_on = id_field)

    custom_dataset, custom_df = prepare_dataset(custom_file, [custom_task_field], custom_file, FP_type,
                                                smiles_field = custom_smi_field, id_field = custom_id_field)

    # step2: calculate transfer value
    print('==== calculating transfer values ... ====')
    transfer_model = calculate_transfer_values(prev_model, n_layer)
    X_new = np.concatenate([custom_dataset.X, dataset.X])
    value_reduced = dim_reduce(transfer_model, X_new)
    
    custom_df['coord1'] = value_reduced[0:len(custom_df),0]
    custom_df['coord2'] = value_reduced[0:len(custom_df),1]
    df['coord1'] = value_reduced[len(custom_df):value_reduced.shape[0],0]
    df['coord2'] = value_reduced[len(custom_df):value_reduced.shape[0],1]

    # step3: MiniBatch clustering
    mbk = cluster_MiniBatch(value_reduced)
    custom_df['Label'] = mbk.labels_[0:len(custom_df)]
    df['Label'] = mbk.labels_[len(custom_df):value_reduced.shape[0]]

    # step4: SAR rendering for all compounds
    print('==== rendering SAR for chemicals on the landscape ... ====')
    custom_df = SAR_rendering(custom_dataset.X, prev_model, custom_df, custom_id_field, custom_smi_field, 
                              custom_SAR_result_dir, vis_cutoff)

    # step5: pack sdf
    if pack_sdf:
        print('==== packing sdf file ... ====')
        plot_df = pd.DataFrame({'coord1': value_reduced[:,0], 'coord2': value_reduced[:,1]})
        plot_df[id_field] = custom_df[custom_id_field].tolist() + df[id_field].tolist()
        plot_df[task_name] = custom_df[custom_task_field].tolist() + df[task_name].tolist()
        plot_df['Label'] = mbk.labels_
        plot_df[smiles_field] = custom_df[custom_smi_field].tolist() + df[smiles_field].tolist()
        plot_df['imgs'] = custom_df['imgs'].tolist() + df['imgs'].tolist()
        df2sdf(plot_df, custom_sdf_name, smiles_field, id_field)

    # step6: build interactive plot using Bokeh
    plot_df = plot_df.drop(['ROMol'], axis = 1)
    plot_df['group'] = [1] * len(custom_df) + [0] * len(df)
    return plot_df



