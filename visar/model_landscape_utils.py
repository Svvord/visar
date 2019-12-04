# -*- coding: utf-8 -*-
"""
Created on Sun June  16 2019

@author: Dingqy
"""
import deepchem as dc

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K

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

#------------------------------------------------
# functions for transfer value calculating (layer1/2)
def ST_model_layer1(n_features, layer_size, pretrained_params,
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5):
    training_dat = Input(shape = (n_features,), dtype = 'float32')
    X = Dense(layer_size[0], activation = 'relu')(training_dat)
    model = Model(training_dat, X)
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
    model = Model(training_dat, X)
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    model.compile(loss = 'mean_squared_error', optimizer=optimizer)
    
    # set parameters
    model.layers[1].set_weights(pretrained_params[0])
    model.layers[2].set_weights(pretrained_params[1])
    return model

#----------------------------------------------
def get_weights_RobustMT(model, layer_variables):
    with model._get_tf("Graph").as_default():
        w1 = model.session.run(layer_variables[0])
        b1 = model.session.run(layer_variables[1])
    return [w1, b1]

def calculate_transfer_values(prev_model, n_tasks, layer_sizes, bypass_layer_sizes, 
                              n_layer = 2, n_features = 2048):
    # load model
    model = dc.models.RobustMultitaskRegressor(n_tasks = n_tasks, n_features = n_features, layer_sizes = layer_sizes,
                                               bypass_layer_sizes= bypass_layer_sizes, bypass_dropouts = [.5],
                                               dropout = 0.5, learning_rate = 0.003)
    model.restore(checkpoint = prev_model)

    # load previous parameters
    tot_layer_variables = model.get_variables()
    param1 = get_weights_RobustMT(model, [tot_layer_variables[0], tot_layer_variables[1]])
    param2 = get_weights_RobustMT(model, [tot_layer_variables[2], tot_layer_variables[3]])
    
    n_features = param1[0].shape[0]
    layer_size = [param1[0].shape[1], param2[0].shape[1]]
    
    if n_layer == 1:
        transfer_model = ST_model_layer1(n_features, layer_size, [param1, param2])
    elif n_layer == 2:
        transfer_model = ST_model_layer2(n_features, layer_size, [param1, param2])
    else:
        print('invalid layer size!')
    return transfer_model

#--------------------------------------------
def calculate_transfer_values_ST(prev_model, n_layer):
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

def cluster_MiniBatch(values, grain_size = 30):
    # clustering using KMeans minibatch algorithm
    n_clusters = int(values.shape[0] / grain_size)
    n_clusters = min(n_clusters, 500)
    mbk = MiniBatchKMeans(init = 'k-means++', init_size = 501, n_clusters = n_clusters, batch_size=100,
                      n_init = 10, max_no_improvement=10, verbose=0, random_state=0)
    mbk.fit(values)
    return mbk

#----------------------------------------------
def df2sdf(df, output_sdf_name, 
           smiles_field = 'canonical_smiles', id_field = 'chembl_id', 
           selected_batch = None):
    '''
    pack pd.DataFrame to sdf_file
    '''
    if not selected_batch is None:
        df = df.loc[df['label'] == selected_batch]
    PandasTools.AddMoleculeColumnToFrame(df,smiles_field,'ROMol')
    PandasTools.WriteSDF(df, output_sdf_name, idName=id_field, properties=df.columns)

    return

