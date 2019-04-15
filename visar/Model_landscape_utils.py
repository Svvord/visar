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
