# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:01:23 2019

@author: Dingqy
"""

from keras import backend as K
import tensorflow as tf
from rdkit.Chem import rdMolDescriptors
from keras.models import load_model
from rdkit import Chem
import numpy as np
import matplotlib.cm as cm
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

from Model_training_utils import optimize_SVR, optimize_RidgeCV

from TRACE_utils import Parameters

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

def calculate_gradients_TRACE(rown, coln, TRACE_param, k):
    '''
    return the coefient matrix of TRACE model
    '''
    parameter_result = Parameters(rown, coln)
    parameter_result.from_ndarray(TRACE_param, rown, coln)
    
    omega_matrix = parameter_result._omega_matrix
    return omega_matrix[k,:].flatten()

def gradient2atom(smi, gradient, pos_cut = 3, neg_cut = -3, nBits = 2048):
    """
    map the gradient of Morgan fingerprint bit on the molecule
    Input:
        smi - the smiles of the molecule (a string)
        gradient - the 2048 coeffients of the feature
        cutoff - if positive, get the pos where the integrated weight is bigger than the cutoff;
                 if negative, get the pos where the integrated weight is smaller than the cutoff
    Output:
        two list of atom ids (positive and negative)   
    """
    # generate mol 
    mol = Chem.MolFromSmiles(smi)
    # get the bit info of the Morgan fingerprint
    bi = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = 2, bitInfo=bi, nBits=nBits)
    onbits = list(fp.GetOnBits())
    # calculate the integrated weight
    atomsToUse = np.zeros((len(mol.GetAtoms()),1))
    for bitId in onbits:
        atomID, radius = bi[bitId][0]
        temp_atomsToUse = []
        if radius > 0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
            for b in env:
                temp_atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
                temp_atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        else:
            temp_atomsToUse.append(atomID)
            env = None
        temp_atomsToUse = list(set(temp_atomsToUse))
        atomsToUse[temp_atomsToUse] += gradient[bitId]
    # get the postively/negatively contributed atom ids
    highlit_pos = []
    highlit_neg = []
    for i in range(len(atomsToUse)):
        if  atomsToUse[i] > pos_cut:
            highlit_pos.append(i)
        elif atomsToUse[i] < neg_cut:
            highlit_neg.append(i)
    return mol, highlit_pos, highlit_neg, atomsToUse

def color_rendering(atomsToUse, cutoff):
    cmap = cm.RdBu_r
    color_dict = {}
    #print(atomsToUse)
    atomsToUse = (atomsToUse.flatten() / cutoff) + 0.5
    for i in range(len(atomsToUse)):
        color_dict[i] = cmap(atomsToUse[i])[0:3]
    return atomsToUse, color_dict

def moltosvg(mol,molSize=(450,200),kekulize=True,drawer=None,**kwargs):
    mc = rdMolDraw2D.PrepareMolForDrawing(mol,kekulize=kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,**kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:',''))
