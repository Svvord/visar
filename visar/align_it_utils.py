#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:07:36 2019

The complete workflow of align-it analysis

@author: dingqy14
"""
import os
import glob
import numpy as np
import pandas as pd
import itertools
import seaborn as sns

def find(lst, a):
    return [i for i, x in enumerate(lst) if x == a]

def parse_phar_file(fname, saving_path):
    '''
    pharse the original phar files and split the content into a series of phar files, each containing only one phar
    '''
    with open(fname, 'rb') as f:
        content = f.readlines()
    
    cnt = 0
    phar_part = []
    name_store = []
    for i in range(len(content)):
        if len(phar_part) == 0:
            name = content[i].strip()
            if name in name_store:
                conf_cnt = len(find(name_store, name))
                name = name + '_conf' + str(conf_cnt+1)
            else:
                name = name + '_conf1'
            phar_part.append(name + '\n')
            name_store.append(content[i].strip())
        
        elif content[i] == '$$$$\n':
            cnt += 1
            phar_part.append(content[i])
            # save the file
            with open(saving_path + str(name) + '.phar', 'wb') as f:
                f.write(''.join(phar_part))
                # clear memory
            phar_part = []
        else:
            phar_part.append(content[i])
    print '%d phar files processed' % cnt
    return 

def parse_sdf_file(fname, saving_path):
    '''
    pharse the original sdf files and split the content into a series of phar files, each containing only one sdf
    '''
    with open(fname, 'rb') as f:
        content = f.readlines()
        
    cnt = 0
    sdf_part = []
    for i in range(len(content)):
        if len(sdf_part) == 0:
            name = content[i].strip()
            sdf_part.append(content[i])
        elif content[i] == '$$$$\n':
            cnt += 1
            sdf_part.append(content[i])
            with open(saving_path + str(name) + '.sdf', 'wb') as f:
                f.write(''.join(sdf_part))
            sdf_part = []
        else:
            sdf_part.append(content[i])
    print '%d sdf files processed' % cnt
    return

def calculate_pair_distance(input_path, saving_path):
    '''
    calculate pairwise distance between phars in the input directory
    '''
    os.mkdir(saving_path)
    fnames = glob.glob(input_path + '*.phar')
    labels = [x.split('.phar')[-2].split('/')[-1] for x in fnames]
    # calculate pair distance by calling align-it
    for i in range(len(fnames)):
        for j in range(len(fnames)):
            if labels[i].split('_')[0] != labels[j].split('_')[0]:
                os.system('align-it --reference %s --refType PHAR --dbase %s \
                          --dbType PHAR --pharmacophore %s --scores %s' 
                          % (fnames[i], fnames[j], saving_path+labels[i]+'-'+labels[j]+'.phar', 
                             saving_path+labels[i]+'-'+labels[j]+'.tab')) 
    return

def clean_dist_mat(unique_labels, labels, raw_dist_mat, dist_type):
    '''
    for the cases of chemicals with multiple conformations, the smallest distance would be kept in the distance matrix.
    input: unique_labels --- a list of names of unique chemicals
           labels --- a list of the actual names of chemicals in the input directory
           raw_dist_mat --- raw distance matrix of all chemicals in the input directory
           dist_type --- one of ('TVERSKY_REF', 'TVERSKY_DB', 'TANIMOTO')
    output: distance matrix of unique chemicals
    '''
    all_labels = [x.split('_')[0] for x in labels]
    max_score_stores = []
    max_pair_stores = []
    phar1 = []
    phar2 = []
    clean_matrix = np.zeros((len(unique_labels), len(unique_labels)))
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            conf1 = find(all_labels, unique_labels[i])
            conf2 = find(all_labels, unique_labels[j])
            max_score = 0
            max_pair = ''
            for (idx1, idx2) in itertools.product(conf1, conf2):
                if raw_dist_mat[idx1, idx2] > max_score:
                    max_score = raw_dist_mat[idx1, idx2]
                    max_pair = labels[idx1] + '-' + labels[idx2]
            max_score_stores.append(max_score)
            max_pair_stores.append(max_pair)
            phar1.append(unique_labels[i])
            phar2.append(unique_labels[j])
            clean_matrix[i,j] = max_score
    clean_matrix = clean_matrix.T + clean_matrix
    info_df = pd.DataFrame({'phar1':phar1, 'phar2':phar2, 
                            'max_score': max_score_stores,
                            'max_pair': max_pair_stores})
    info_df.to_csv('./' + dist_type +'_dist.csv')
    df = pd.DataFrame(clean_matrix)
    df.index = unique_labels
    df.columns = unique_labels
    return df

def generate_rep_PHAR(phar_saving_path, dist_saving_path, sdf_file, 
                      output_name, saving_path):
    '''
    generate representative pharmacophore of the group of chemicals
    '''
    fnames = glob.glob(phar_saving_path + '*.phar')
    labels = [x.split('.phar')[-2].split('/')[-1] for x in fnames]
    all_labels = [x.split('_')[0] for x in labels]
    
    # extract distance from temp files
    TANIMOTO_mat = np.zeros((len(fnames), len(fnames)))
    TVERSKY_REF_mat = np.zeros((len(fnames), len(fnames)))
    TVERSKY_DB_mat = np.zeros((len(fnames), len(fnames)))
    for i in range(len(fnames)):
        for j in range(len(fnames)):
            if labels[i].split('_')[0] != labels[j].split('_')[0]:
                query_name = dist_saving_path+labels[i]+'-'+labels[j]+'.tab'
                with open(query_name,'rb') as f:
                    content = f.readlines()
                    content = content[0]
                    TANIMOTO_mat[i,j] = content.strip().split('\t')[8]
                    TVERSKY_REF_mat[i,j] = content.strip().split('\t')[9]
                    TVERSKY_DB_mat[i,j] = content.strip().split('\t')[10]
    
    unique_labels = list(set([x.split('_')[0] for x in labels]))
    
    # clean-up and dealing with multiple conformation
    TVEREF_mat = clean_dist_mat(unique_labels, labels, TVERSKY_REF_mat, 'TVERSKY_REF')
    TVEDB_mat = clean_dist_mat(unique_labels, labels, TVERSKY_DB_mat, 'TVERSKY_DB')
    TANI_mat = clean_dist_mat(unique_labels, labels, TANIMOTO_mat, 'TANIMOTO')
    
    sns_plot = sns.clustermap(TVEREF_mat)
    sns_plot.savefig(saving_path + "TVEREF_mat.png")
    sns_plot = sns.clustermap(TVEDB_mat)
    sns_plot.savefig(saving_path + "TVEDB_mat.png")
    sns_plot = sns.clustermap(TANI_mat)
    sns_plot.savefig(saving_path + "TANI_mat.png")
    
    # select representative phars and generate vis file
    col_avg_TANI = (np.sum(TANI_mat, axis = 0) / TANI_mat.shape[0]).tolist()
    col_avg_TVEREF = (np.sum(TVEREF_mat, axis = 0) / TVEREF_mat.shape[0]).tolist()
    col_avg_TVEDB = (np.sum(TVEDB_mat, axis = 0) / TVEDB_mat.shape[0]).tolist()
    col_avg = np.array(col_avg_TANI) + np.array(col_avg_TVEREF) + np.array(col_avg_TVEDB)
    
    select_phar = [unique_labels[m] for m in find(col_avg, max(col_avg))][0]
    
    # calculate the aligned conformation between select_phar and all molecules
    select_phar_file = [labels[m] for m in find(all_labels, select_phar)]
    os.mkdir('./SDF_store/')
    for i in range(len(select_phar_file)):
        os.system('align-it --reference %s --refType PHAR --dbase %s \
                  --dbType SDF --pharmacophore %s --scores %s \
                  --rankBy TVERSKY_REF --out %s' 
                  % (phar_saving_path+select_phar_file[i]+'.phar', 
                     sdf_file, 
                     saving_path+output_name+str(i)+'.phar', 
                     saving_path+output_name+str(i)+'.tab', 
                     saving_path+output_name+str(i)+'.sdf'
                     )) 
        os.system('cp %s %s' % (phar_saving_path+select_phar_file[i]+'.phar', 
                                saving_path+select_phar_file[i]+'.phar'))
        parse_sdf_file(saving_path+output_name+str(i)+'.sdf', './SDF_store/Conf' + str(i) + '_')
    return

#=============================
def proceed_pharmacophore(home_dir, sdf_file, result_dir, output_name):
    '''
    the complete workflow of pharmacophore generation
    '''
    # set environment and params
    os.system('mkdir %s' % result_dir)
    os.chdir(result_dir)

    # generate pharmacophore and parse them into individual files
    fname = result_dir + 'align-it.phar'
    os.system('align-it --dbase %s --dbType SDF --pharmacophore %s' % 
              (sdf_file, fname))
    phar_saving_path = result_dir + 'temp_phars/'
    os.system('mkdir %s' % phar_saving_path)
    parse_phar_file(fname, phar_saving_path)

    # caculate the distance between each pair 
    dist_saving_path = result_dir + 'temp_dist/'
    calculate_pair_distance(phar_saving_path, dist_saving_path)

    # generate representative phar files and clear the temp files    
    saving_path = './'
    generate_rep_PHAR(phar_saving_path, dist_saving_path, sdf_file, output_name, saving_path)
    os.system('rm -r %s' % phar_saving_path)
    os.system('rm -r %s' % dist_saving_path)
    return


###############################
'''
home_dir = '/Users/dingqy14/Desktop/Pharma_Viz/rdkit-codes/data/'
os.chdir(home_dir)

# prepare ligand conformations
from rdkit import Chem
from rdkit.Chem import AllChem

raw_sdf_file = 'Label_9.sdf'
sdf_file = home_dir + 'Label9_rdkit_conf.sdf'
ms = [x for x in Chem.SDMolSupplier(raw_sdf_file)]
n_conf = 5
w = Chem.SDWriter(sdf_file)
for i in range(n_conf):
    ms_addH = [Chem.AddHs(m) for m in ms]
    for m in ms_addH:
        AllChem.EmbedMolecule(m)
        AllChem.MMFFOptimizeMoleculeConfs(m)
        w.write(m)

# process pharmacophores
result_dir = home_dir + 'Label9_rdkit_phars/'
output_name = 'Cluster9_'
proceed_pharmacophore(home_dir, sdf_file, result_dir, output_name)
'''
