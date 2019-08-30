# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:34:48 2019

@author: Dingqy
"""

import numpy as np
import pandas as pd
from scipy import optimize
import math
import pickle

import deepchem as dc
from Model_training_utils import prepare_dataset

from math import sqrt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

class Parameters(object):
    init_mean = 0.05
    init_sigma = 1

    def __init__(self, rown, coln):
        self._omega_matrix = self.init_sigma * \
                             np.random.randn(rown, coln) + self.init_mean
        self._omega_mean = np.zeros(coln, dtype=np.float32) + self.init_mean
        self._sigma_in_colums = np.zeros(
            coln, dtype=np.float32) + self.init_sigma
        self._sigma_in_group = np.zeros(
            rown, dtype=np.float32) + self.init_sigma

    def to_ndarray(self, rown, coln):
        result = np.zeros((rown + 2, coln + 1), np.float32)
        matrix_size = rown * coln
        result.put(range(matrix_size), self._omega_matrix)
        result.put(range(matrix_size, matrix_size + coln), self._omega_mean)
        result.put(range(matrix_size + coln, matrix_size + 2 * coln), self._sigma_in_colums)
        result.put(range(matrix_size + 2 * coln, matrix_size + 2 * coln + rown), self._sigma_in_group)
        #result.flat[range(matrix_size, matrix_size + coln)] = self._omega_mean
        #result.flat[range(matrix_size + coln, matrix_size +
        #                  2 * coln)] = self._sigma_in_colums
        #result.flat[range(matrix_size + 2 * coln, matrix_size + 2 * coln +
        #                  rown)] = self._sigma_in_group
        return result

    def from_ndarray(self, ndarray, rown, coln):
        if ndarray.size < (rown + 2) * (coln + 1):
            raise RuntimeError("ndarray size is not enough.")
        matrix_size = rown * coln
        self._omega_matrix.put(range(matrix_size), ndarray)      
        self._omega_mean = ndarray.flat[[x for x in range(matrix_size, matrix_size + coln)]]
        self._sigma_in_colums = ndarray.flat[[x for x in range(matrix_size + coln,
                                                   matrix_size + 2 * coln)]]
        self._sigma_in_group = ndarray.flat[[ x for x in range(
            matrix_size + 2 * coln, matrix_size + 2 * coln + rown)]]    
    
class TRACE(object):

    def __init__(self, rown, coln, proteins, X_train, Y_train, X_test, 
                 Y_test, df, scale_g, scale_c):
        self._rown = rown
        self._coln = coln
        self._proteins = proteins
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        self.f_value = []
        
        self.df = df
        self.scale_group = scale_g
        self.scale_coln = scale_c
        
        N_train_list = []
        for i in range(rown):
            N_train_list.append(X_train[i].shape[0])
        self.N_train = N_train_list

    def loss_fun(self, ndarray):
        rown = self._rown
        coln = self._coln
        # genearte paramters of correct format
        parameter = Parameters(rown, coln)
        parameter.from_ndarray(ndarray, rown, coln)
            
        # 1. summation of likelihood function of each protein.
        tmp1 = sum(rown/2 * np.array([math.log(2 * math.pi * x) for x in parameter._sigma_in_colums]))
        omega_rep = np.matrix(parameter._omega_mean.repeat(rown, axis = 0)).reshape(coln, rown)
        delta_omega = parameter._omega_matrix - omega_rep.T
        for i in range(coln):
            tmp1 += (delta_omega[:,i].T * delta_omega[:,i]) / (2 * parameter._sigma_in_colums[i])
        # 2. summation of log format of prior distribution of sigma_y for each
        #    protein(regression epsilon's variance)
        tmp2 = sum([self.N_train[j]/2 * math.log(2* math.pi * parameter._sigma_in_group[j]) for j in range(rown)])        
        for i in range(rown):
            tmp_diff = self.Y_train[i] - np.matrix(parameter._omega_matrix[i,:]) * self.X_train[i].T 
            tmp2 += np.inner(tmp_diff, tmp_diff) / (2* parameter._sigma_in_group[i])
        # 3. summation of log format of prior distribution of omega for each
        #    protein.
        tmp3 = sum([math.log(x) / (1 + self.df/2) + self.df * self.scale_coln / (2*x) for x in parameter._sigma_in_colums])
        # 4. log format of prior distribution of variance of omega for each
        #    colum of omega matrix.
        tmp4 = sum([math.log(x) / (1 + self.df/2) + self.df * self.scale_group / (2*x) for x in parameter._sigma_in_group])
        value = float(tmp1 + tmp2 + tmp3 + tmp4)
        self.f_value.append(value)
        return value
    
    def fprime(self, p):
        rown = self._rown
        coln = self._coln
        # genearte paramters of correct format
        parameter = Parameters(rown, coln)
        parameter.from_ndarray(p, rown, coln)
        output = np.zeros(p.shape)
        matrix_size = coln * rown
        
        #1 fprime of omega_matrix
        ## calculate row by row (iterate over each protein of the family)
        for i in range(rown):
            term1 = (parameter._omega_matrix[i,:] - parameter._omega_mean) / parameter._sigma_in_colums
            tmp_diff = self.Y_train[i] - np.matrix(parameter._omega_matrix[i,:]) * self.X_train[i].T
            term = term1 - 2 * tmp_diff * self.X_train[i]
            output.flat[[x for x in range(i * coln, (i+1) * coln)]] = term
        
        #2 fprime of omega_mean
        omega_rep = np.matrix(parameter._omega_mean.repeat(rown, axis = 0)).reshape(coln, rown)
        delta_omega = parameter._omega_matrix - omega_rep.T
        output.flat[[x for x in range(matrix_size, matrix_size + coln)]] = - np.sum(delta_omega, axis = 0) / parameter._sigma_in_colums
        
        #3 fprime of sigma in column
        term3 = (1 / (1+ self.df/2) + rown / 2) * (1 / parameter._sigma_in_colums)
        term4 = np.array([(np.inner(delta_omega[:,j].T, delta_omega[:,j].T) + self.df*self.scale_coln) /(2 * parameter._sigma_in_colums[j]**2) for j in range(coln)]).flatten()
        output.flat[[x for x in range(matrix_size + coln, matrix_size + 2* coln)]] = term3 - term4
        
        #4 fprime of sigma in group
        term5 = (1/(1+self.df/2) + np.array(self.N_train) / 2) * (1 / parameter._sigma_in_group)
        for j in range(rown):
            tmp_diff = self.Y_train[j] - np.matrix(parameter._omega_matrix[j,:]) * self.X_train[j].T
            term6 = (np.array(np.inner(tmp_diff, tmp_diff)  + self.df*self.scale_group) / (2 * parameter._sigma_in_group[j]**2))[0]
            output.flat[matrix_size + 2* coln + j] = term5[j] - term6
        return output
    
    def set_bound(self):
        # * (min, max) for each x
        rown = self._rown
        coln = self._coln
        
        matrix_size = rown * coln
        bnds = [(None, None) for i in range(matrix_size + coln)] + \
           [(0.01, None) for i in range(matrix_size + coln, matrix_size + 2 * coln + rown)] + \
           [(0,0), (0,0)]
        bnds = tuple(x for x in bnds)
        return bnds

def fmin_trace(target, MAX_ITERATION = 15000):
    OPT_METHOD = "L-BFGS-B"
    #MAX_ITERATION = 15000
    
    init_point = Parameters(target._rown, target._coln).to_ndarray(target._rown, target._coln)
    res = optimize.minimize(target.loss_fun, init_point, method = OPT_METHOD,
                            jac = target.fprime,
                            hess=False,
                            bounds=target.set_bound(),
                            options={'maxiter': MAX_ITERATION,
                                     'disp': True,
                                     'maxfun': 15000})
    return res, target.f_value


def TRACE_predict(Xj, trace_obj, param_values, j):
    rown = trace_obj._rown
    coln = trace_obj._coln
    parameter_result = Parameters(rown, coln)
    parameter_result.from_ndarray(param_values, rown, coln)
    omega_matrix = parameter_result._omega_matrix
    y_pred = np.array(np.matrix(omega_matrix[j,:]) * Xj.T)[0]
    return y_pred

def prepare_TRACE_dataset(fname, task_names, FP_type, result_path):
    X_train = []
    X_test = []
    X_val = []
    Y_train = []
    Y_test = []
    Y_val = []
    for task in task_names:
        print('----- Processing dataset of %s -------' % task)
        dataset_file = '%s/temp.csv' % (result_path)
        dataset = prepare_dataset(fname, [task], dataset_file, FP_type)
        
        for rep in range(3):
            splitter = dc.splits.RandomSplitter(dataset_file)
            train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)
        
            X_train.append(np.c_[[1]*train_dataset.X.shape[0], train_dataset.X])
            X_test.append(np.c_[[1]*test_dataset.X.shape[0], test_dataset.X])
            X_val.append(np.c_[[1]*valid_dataset.X.shape[0], valid_dataset.X])
            Y_train.append(train_dataset.y.flatten())
            Y_test.append(test_dataset.y.flatten())
            Y_val.append(valid_dataset.y.flatten())
    
    for rep in range(3):
        X_train0 = [X_train[k] for k in range(len(X_train)) if k % 3 == rep]
        X_test0 = [X_test[k] for k in range(len(X_train)) if k % 3 == rep]
        X_val0 = [X_val[k] for k in range(len(X_train)) if k % 3 == rep]
        Y_train0 = [Y_train[k] for k in range(len(X_train)) if k % 3 == rep]
        Y_test0 = [Y_test[k] for k in range(len(X_train)) if k % 3 == rep]
        Y_val0 = [Y_val[k] for k in range(len(X_train)) if k % 3 == rep]
        pickle.dump([X_train0, X_test0, X_val0, Y_train0, Y_test0, Y_val0], 
                    open(result_path + '/temp_%d_dataset.p' % (rep), 'wb'))
    return

def TRACE_model_hyperparam_screen(fname, task_names, FP_type, hyperparams, 
                                  MAX_ITERATION = 100, log_path = './logs/'):
    log_output = []
    output_store = []
    hyperparam_record = []
    for k in range(len(hyperparams)):
        for rep in range(3):
            print('---- processing hyperparam %s of rep %d ----' % (str(hyperparams[k]), rep))
            [X_train,_,X_val,Y_train,_,Y_val]=pickle.load(open(log_path+'/temp_%d_dataset.p'%(rep),'rb'))
            rown = len(X_train)
            coln = X_train[0].shape[1]
            output_dat = np.zeros((rown,2))
            df = hyperparams[k][0]
            scale_g = hyperparams[k][1]
            scale_c = hyperparams[k][2]
            trace_obj = TRACE(rown, coln, task_names, X_train, Y_train, X_val, Y_val, 
                  df, scale_g, scale_c)
            res, values = fmin_trace(trace_obj, MAX_ITERATION)
            for j in range(rown):
                y_pred = TRACE_predict(X_val[j], trace_obj, res['x'], j)
                y_real = Y_val[j]
                output_dat[j,0] = sqrt(mean_squared_error(y_pred, y_real))
                output_dat[j,1] = pearsonr(y_pred, y_real)[0]
            output_store.append(np.array(output_dat))
        log_output.append(np.concatenate(output_store, axis = 1))
        output_store = []
        hyperparam_record = hyperparam_record + [str(hyperparams[k]) for _ in range(rown)]
        out = np.concatenate(log_output, axis = 0)
        df = pd.DataFrame(out)
        df['label'] = hyperparam_record
        df.to_csv(log_path + 'hyperparam_screen.csv')
    return df

def TRACE_model_training(task_names, log_path, params, output_prefix):
    task_label = ['' for _ in range(len(task_names) * 3)]
    rep_label = ['' for _ in range(len(task_names) * 3)]
    output_dat = np.zeros((len(task_names) * 3 ,4))
    cnt = 0
    for rep in range(3):
        print('---- processing rep %d ----' % (rep))
        [X_train,_,X_test,Y_train,_,Y_test]=pickle.load(open(log_path+'/temp_%d_dataset.p'%(rep),'rb'))
        rown = len(X_train)
        coln = X_train[0].shape[1]
        df = params[0]
        scale_g = params[1]
        scale_c = params[2]
        trace_obj = TRACE(rown, coln, task_names, X_train, Y_train, X_test, Y_test, 
              df, scale_g, scale_c)
        res, values = fmin_trace(trace_obj)
        for j in range(rown):
            y_pred = TRACE_predict(X_train[j], trace_obj, res['x'], j)
            y_real = Y_train[j]
            output_dat[cnt,0] = sqrt(mean_squared_error(y_pred, y_real))
            output_dat[cnt,1] = pearsonr(y_pred, y_real)[0]
            
            y_pred = TRACE_predict(X_test[j], trace_obj, res['x'], j)
            y_real = Y_test[j]
            output_dat[cnt,2] = sqrt(mean_squared_error(y_pred, y_real))
            output_dat[cnt,3] = pearsonr(y_pred, y_real)[0]
            task_label[cnt] = task_names[j]
            rep_label[cnt] = 'rep' + str(rep)
            cnt += 1
            
        df = pd.DataFrame(output_dat)
        df['task'] = task_label
        df['rep'] = rep_label
        df.to_csv(log_path + output_prefix + '.csv')
        
        pickle.dump(res['x'], open(output_prefix + '_' + str(rep) + '.p', 'wb'))

