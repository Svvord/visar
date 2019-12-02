# -*- coding: utf-8 -*-
"""
Created on Sun June  16 2019

@author: Dingqy
"""

import tensorflow.keras.backend as K
import tensorflow as tf
from rdkit.Chem import rdMolDescriptors
from tensorflow.keras.models import load_model
from rdkit import Chem
import numpy as np
import matplotlib.cm as cm
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

from visar.model_training_utils import optimize_SVR, optimize_RidgeCV

# gradient calculation
def calculate_gradients_RobustMT(X_train, task_tensor_name, prev_model,
								 n_tasks, n_features, layer_sizes, bypass_layer_sizes):
    '''
    Calculate the gradients for each chemical
    input: X_train --- fingerprint matrix of the chemicals of interest
           prev_model -- trained neural network model
    output: the gradient matrix
    '''
    feed_dict = {}

    with tf.Graph().as_default():
        with tf.Session() as sess:
            K.set_session(sess)

            new_saver = tf.train.import_meta_graph(prev_model + '.meta')
            new_saver.restore(sess, prev_model)
            graph = tf.get_default_graph()

            feed_dict['Feature_8/PlaceholderWithDefault:0'] = X_train
            #feed_dict['Dense_7/Dense_7/Relu:0'] = X_train[0:10,0:512]
            feed_dict['Placeholder:0'] = 1.0

            op_tensor = graph.get_tensor_by_name(task_tensor_name)
            X = graph.get_tensor_by_name('Feature_8/PlaceholderWithDefault:0')
            #X = graph.get_tensor_by_name('Dense_7/Dense_7/Relu:0')

            reconstruct = tf.gradients(op_tensor, X)[0]
            out = sess.run(reconstruct, feed_dict = feed_dict)[0]

    K.clear_session()
    return out

def calculate_gradients_ST(X_train, prev_model):
    '''
    Calculate the gradients for each chemical
    input: X_train --- fingerprint matrix of the chemicals of interest
           prev_model -- trained neural network model
    output: the gradient matrix
    '''
    feed_dict = {}

    with tf.Graph().as_default():
        with tf.Session() as sess:
            K.set_session(sess)
            model = load_model(prev_model)
            feed_dict[model.input.name] = X_train

            layer = model.layers[1]
            op_tensor = layer.output
            X = model.layers[0].input
            reconstruct = tf.gradients(op_tensor, X)[0]

            out = sess.run(reconstruct, feed_dict = feed_dict)
    K.clear_session()
    return out

def calculate_gradients_baseline(train_dataset, model_type):
    '''
    return the coefients of baseline model after training
    Notice the input dataset must be training dataset!
    '''
    X_new = np.c_[[1]*train_dataset.X.shape[0], train_dataset.X]
    if model_type == 'SVR':
        model = optimize_SVR(X_new, train_dataset.y)
        out = model.coef_
    elif model_type == 'RidgeCV':
        model = optimize_RidgeCV(X_new, train_dataset.y)
        out = model.coef_.flatten()
    else:
        print('model type not supported!')
        return
    return out
